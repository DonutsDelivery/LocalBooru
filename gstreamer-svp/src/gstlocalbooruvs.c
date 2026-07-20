#include <gst/gst.h>
#include <gst/video/video.h>
#include <vapoursynth/VSScript4.h>

#ifndef PACKAGE
#define PACKAGE "localbooru-gst-vapoursynth"
#endif

#define DEFAULT_MAX_REQUESTS 25
#define DEFAULT_MAX_BUFFERED 32
#define MIN_CACHE_BYTES (256LL * 1024 * 1024)
#define MAX_CACHE_BYTES (1024LL * 1024 * 1024)

static gchar *get_script_path(void) {
    const char *direct_path = g_getenv("LOCALBOORU_VS_SCRIPT");
    if (direct_path && *direct_path)
        return g_strdup(direct_path);

    const char *path_file = g_getenv("LOCALBOORU_VS_SCRIPT_FILE");
    gchar *contents = NULL;
    if (path_file && *path_file && g_file_get_contents(path_file, &contents, NULL, NULL)) {
        g_strstrip(contents);
        if (*contents)
            return contents;
        g_free(contents);
    }
    return NULL;
}

typedef struct {
    GstBuffer *buffer;
    GstClockTime pts;
    GstClockTime duration;
} InputFrame;

typedef struct {
    GstElement parent;
    GstPad *sink_pad;
    GstPad *src_pad;

    GMutex lock;
    GCond input_ready;
    GCond output_ready;
    GQueue inputs;
    gint64 input_start;
    gint64 input_next;
    guint max_buffered;
    gboolean shutdown;
    gboolean eof;
    gboolean passthrough;
    GstFlowReturn downstream_flow;

    GstVideoInfo input_info;
    GstVideoInfo output_info;
    GstCaps *input_caps;
    GstClockTime first_pts;
    GstClockTime output_duration;
    gboolean first_output;

    const VSSCRIPTAPI *vssapi;
    const VSAPI *vsapi;
    VSScript *script;
    VSCore *core;
    VSNode *input_node;
    VSNode *output_node;
    GThread *worker;
    const VSFrame *output_frames[DEFAULT_MAX_REQUESTS];
    gchar *output_errors[DEFAULT_MAX_REQUESTS];
    gboolean output_completed[DEFAULT_MAX_REQUESTS];
    guint pending_requests;
    gint64 output_start;
} GstLocalBooruVS;

typedef struct {
    GstElementClass parent_class;
} GstLocalBooruVSClass;

G_DEFINE_TYPE(GstLocalBooruVS, gst_localbooru_vs, GST_TYPE_ELEMENT)

GST_DEBUG_CATEGORY_STATIC(localbooru_vs_debug);
#define GST_CAT_DEFAULT localbooru_vs_debug

static void input_frame_free(InputFrame *frame) {
    if (!frame)
        return;
    gst_buffer_unref(frame->buffer);
    g_free(frame);
}

static void clear_input_queue_locked(GstLocalBooruVS *self) {
    while (!g_queue_is_empty(&self->inputs))
        input_frame_free(g_queue_pop_head(&self->inputs));
    self->input_start = 0;
    self->input_next = 0;
    self->first_pts = GST_CLOCK_TIME_NONE;
    self->first_output = TRUE;
}

static gboolean copy_gst_to_vs(GstLocalBooruVS *self, const InputFrame *input, VSFrame *output) {
    GstVideoFrame frame;
    if (!gst_video_frame_map(&frame, &self->input_info, input->buffer, GST_MAP_READ))
        return FALSE;

    gboolean ok = TRUE;
    for (guint plane = 0; plane < 3; plane++) {
        const guint8 *src = GST_VIDEO_FRAME_PLANE_DATA(&frame, plane);
        const gint src_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, plane);
        guint8 *dst = self->vsapi->getWritePtr(output, plane);
        const gssize dst_stride = self->vsapi->getStride(output, plane);
        const gint width = self->vsapi->getFrameWidth(output, plane);
        const gint height = self->vsapi->getFrameHeight(output, plane);
        if (!src || !dst || width <= 0 || height <= 0) {
            ok = FALSE;
            break;
        }
        for (gint row = 0; row < height; row++)
            memcpy(dst + row * dst_stride, src + row * src_stride, (gsize)width);
    }
    gst_video_frame_unmap(&frame);

    if (ok) {
        VSMap *props = self->vsapi->getFramePropertiesRW(output);
        GstClockTime duration = GST_CLOCK_TIME_IS_VALID(input->duration)
            ? input->duration
            : gst_util_uint64_scale_int(GST_SECOND,
                                        GST_VIDEO_INFO_FPS_D(&self->input_info),
                                        GST_VIDEO_INFO_FPS_N(&self->input_info));
        self->vsapi->mapSetInt(props, "_DurationNum", (gint64)duration, maReplace);
        self->vsapi->mapSetInt(props, "_DurationDen", GST_SECOND, maReplace);
    }
    return ok;
}

static const VSFrame *VS_CC input_get_frame(int frame_number, int activation_reason,
                                             void *instance_data, void **frame_data,
                                             VSFrameContext *frame_ctx, VSCore *core,
                                             const VSAPI *vsapi) {
    (void)activation_reason;
    (void)frame_data;
    (void)core;
    GstLocalBooruVS *self = instance_data;
    InputFrame *input = NULL;

    g_mutex_lock(&self->lock);
    while (!self->shutdown) {
        while (frame_number >= self->input_start + (gint64)self->max_buffered &&
               !g_queue_is_empty(&self->inputs)) {
            input_frame_free(g_queue_pop_head(&self->inputs));
            self->input_start++;
            g_cond_broadcast(&self->input_ready);
        }

        if (frame_number < self->input_start) {
            vsapi->setFilterError("LocalBooruVS: requested frame has left the input buffer", frame_ctx);
            break;
        }

        gint64 offset = frame_number - self->input_start;
        if (offset < (gint64)g_queue_get_length(&self->inputs)) {
            input = g_queue_peek_nth(&self->inputs, (guint)offset);
            break;
        }

        if (self->eof) {
            vsapi->setFilterError("LocalBooruVS: end of input", frame_ctx);
            break;
        }
        g_cond_wait(&self->input_ready, &self->lock);
    }

    if (self->shutdown)
        vsapi->setFilterError("LocalBooruVS: filter reset", frame_ctx);

    VSFrame *result = NULL;
    if (input) {
        VSVideoFormat format;
        if (vsapi->getVideoFormatByID(&format, pfYUV420P8, self->core)) {
            result = vsapi->newVideoFrame(&format,
                                         GST_VIDEO_INFO_WIDTH(&self->input_info),
                                         GST_VIDEO_INFO_HEIGHT(&self->input_info),
                                         NULL, self->core);
            if (result && !copy_gst_to_vs(self, input, result)) {
                vsapi->freeFrame(result);
                result = NULL;
                vsapi->setFilterError("LocalBooruVS: failed to map input frame", frame_ctx);
            }
        }
    }
    g_mutex_unlock(&self->lock);
    return result;
}

static void VS_CC input_free(void *instance_data, VSCore *core, const VSAPI *vsapi) {
    (void)instance_data;
    (void)core;
    (void)vsapi;
}

static GstBuffer *copy_vs_to_gst(GstLocalBooruVS *self, const VSFrame *input,
                                  gint64 frame_number) {
    GstBuffer *buffer = gst_buffer_new_allocate(NULL, GST_VIDEO_INFO_SIZE(&self->output_info), NULL);
    if (!buffer)
        return NULL;

    GstVideoFrame frame;
    if (!gst_video_frame_map(&frame, &self->output_info, buffer, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        return NULL;
    }

    gboolean ok = TRUE;
    for (guint plane = 0; plane < 3; plane++) {
        const guint8 *src = self->vsapi->getReadPtr(input, plane);
        const gssize src_stride = self->vsapi->getStride(input, plane);
        guint8 *dst = GST_VIDEO_FRAME_PLANE_DATA(&frame, plane);
        const gint dst_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, plane);
        const gint width = self->vsapi->getFrameWidth(input, plane);
        const gint height = self->vsapi->getFrameHeight(input, plane);
        if (!src || !dst || width <= 0 || height <= 0) {
            ok = FALSE;
            break;
        }
        for (gint row = 0; row < height; row++)
            memcpy(dst + row * dst_stride, src + row * src_stride, (gsize)width);
    }
    gst_video_frame_unmap(&frame);

    if (!ok) {
        gst_buffer_unref(buffer);
        return NULL;
    }

    g_mutex_lock(&self->lock);
    GstClockTime first_pts = self->first_pts;
    gboolean discont = self->first_output;
    self->first_output = FALSE;
    g_mutex_unlock(&self->lock);

    if (!GST_CLOCK_TIME_IS_VALID(first_pts))
        first_pts = 0;
    GST_BUFFER_PTS(buffer) = first_pts + self->output_duration * frame_number;
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION(buffer) = self->output_duration;
    GST_BUFFER_OFFSET(buffer) = frame_number;
    GST_BUFFER_OFFSET_END(buffer) = frame_number + 1;
    if (discont)
        GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DISCONT);
    return buffer;
}

static void VS_CC output_frame_done(void *user_data, const VSFrame *frame,
                                    int frame_number, VSNode *node,
                                    const char *error_message) {
    (void)node;
    GstLocalBooruVS *self = user_data;
    gboolean discard = FALSE;

    g_mutex_lock(&self->lock);
    if (self->pending_requests > 0)
        self->pending_requests--;
    gint64 index = frame_number - self->output_start;
    if (self->shutdown || index < 0 || index >= DEFAULT_MAX_REQUESTS) {
        discard = TRUE;
    } else {
        self->output_frames[index] = frame;
        self->output_errors[index] = error_message ? g_strdup(error_message) : NULL;
        self->output_completed[index] = TRUE;
    }
    g_cond_broadcast(&self->output_ready);
    g_mutex_unlock(&self->lock);

    if (discard && frame)
        self->vsapi->freeFrame(frame);
}

static void request_output_frame(GstLocalBooruVS *self, gint64 frame_number) {
    g_mutex_lock(&self->lock);
    if (self->shutdown) {
        g_mutex_unlock(&self->lock);
        return;
    }
    self->pending_requests++;
    g_mutex_unlock(&self->lock);
    self->vsapi->getFrameAsync((int)frame_number, self->output_node,
                               output_frame_done, self);
}

static void clear_output_results(GstLocalBooruVS *self) {
    for (guint index = 0; index < DEFAULT_MAX_REQUESTS; index++) {
        if (self->output_frames[index])
            self->vsapi->freeFrame(self->output_frames[index]);
        self->output_frames[index] = NULL;
        g_clear_pointer(&self->output_errors[index], g_free);
        self->output_completed[index] = FALSE;
    }
}

static gpointer output_worker(gpointer data) {
    GstLocalBooruVS *self = data;
    gboolean reached_eos = FALSE;

    for (guint index = 0; index < DEFAULT_MAX_REQUESTS; index++)
        request_output_frame(self, index);

    while (TRUE) {
        g_mutex_lock(&self->lock);
        while (!self->shutdown && !self->output_completed[0])
            g_cond_wait(&self->output_ready, &self->lock);
        if (self->shutdown) {
            g_mutex_unlock(&self->lock);
            break;
        }

        const VSFrame *frame = self->output_frames[0];
        gchar *error = self->output_errors[0];
        gint64 frame_number = self->output_start;
        for (guint index = 0; index + 1 < DEFAULT_MAX_REQUESTS; index++) {
            self->output_frames[index] = self->output_frames[index + 1];
            self->output_errors[index] = self->output_errors[index + 1];
            self->output_completed[index] = self->output_completed[index + 1];
        }
        self->output_frames[DEFAULT_MAX_REQUESTS - 1] = NULL;
        self->output_errors[DEFAULT_MAX_REQUESTS - 1] = NULL;
        self->output_completed[DEFAULT_MAX_REQUESTS - 1] = FALSE;
        self->output_start++;
        gboolean eof = self->eof;
        g_mutex_unlock(&self->lock);

        if (!frame) {
            reached_eos = eof;
            if (!eof) {
                GST_ERROR_OBJECT(self, "VapourSynth output failed: %s",
                                 error ? error : "unknown error");
                g_mutex_lock(&self->lock);
                self->passthrough = TRUE;
                self->shutdown = TRUE;
                g_cond_broadcast(&self->input_ready);
                g_mutex_unlock(&self->lock);
            }
            g_free(error);
            break;
        }
        g_free(error);

        GstBuffer *buffer = copy_vs_to_gst(self, frame, frame_number);
        self->vsapi->freeFrame(frame);
        if (!buffer) {
            GST_ERROR_OBJECT(self, "Failed to allocate VapourSynth output buffer");
            g_mutex_lock(&self->lock);
            self->passthrough = TRUE;
            self->shutdown = TRUE;
            g_cond_broadcast(&self->input_ready);
            g_mutex_unlock(&self->lock);
            break;
        }

        GstFlowReturn flow = gst_pad_push(self->src_pad, buffer);
        if (flow != GST_FLOW_OK) {
            g_mutex_lock(&self->lock);
            self->downstream_flow = flow;
            self->shutdown = TRUE;
            g_cond_broadcast(&self->input_ready);
            g_cond_broadcast(&self->output_ready);
            g_mutex_unlock(&self->lock);
            break;
        }

        request_output_frame(self, frame_number + DEFAULT_MAX_REQUESTS);
    }

    g_mutex_lock(&self->lock);
    while (self->pending_requests > 0)
        g_cond_wait(&self->output_ready, &self->lock);
    gboolean send_eos = reached_eos && !self->shutdown;
    g_mutex_unlock(&self->lock);

    clear_output_results(self);
    g_mutex_lock(&self->lock);
    gboolean switch_to_passthrough = self->passthrough;
    if (switch_to_passthrough)
        clear_input_queue_locked(self);
    g_mutex_unlock(&self->lock);
    if (switch_to_passthrough && self->input_caps)
        gst_pad_push_event(self->src_pad, gst_event_new_caps(self->input_caps));
    if (send_eos)
        gst_pad_push_event(self->src_pad, gst_event_new_eos());
    return NULL;
}

static void destroy_graph(GstLocalBooruVS *self) {
    g_mutex_lock(&self->lock);
    self->shutdown = TRUE;
    g_cond_broadcast(&self->input_ready);
    g_cond_broadcast(&self->output_ready);
    GThread *worker = self->worker;
    self->worker = NULL;
    g_mutex_unlock(&self->lock);

    if (worker && worker != g_thread_self())
        g_thread_join(worker);

    if (self->output_node)
        self->vsapi->freeNode(self->output_node);
    if (self->input_node)
        self->vsapi->freeNode(self->input_node);
    self->output_node = NULL;
    self->input_node = NULL;

    if (self->script)
        self->vssapi->freeScript(self->script);
    self->script = NULL;
    self->core = NULL;
    self->vsapi = NULL;
    self->vssapi = NULL;

    g_mutex_lock(&self->lock);
    clear_input_queue_locked(self);
    self->shutdown = FALSE;
    self->eof = FALSE;
    self->downstream_flow = GST_FLOW_OK;
    self->pending_requests = 0;
    self->output_start = 0;
    g_mutex_unlock(&self->lock);
}

static gboolean create_graph(GstLocalBooruVS *self, GstCaps **output_caps) {
    destroy_graph(self);

    self->vssapi = getVSScriptAPI(VSSCRIPT_API_VERSION);
    if (!self->vssapi) {
        GST_ERROR_OBJECT(self, "VSScript API 4 is unavailable");
        return FALSE;
    }
    self->vsapi = self->vssapi->getVSAPI(VAPOURSYNTH_API_VERSION);
    self->script = self->vssapi->createScript(NULL);
    if (!self->vsapi || !self->script) {
        GST_ERROR_OBJECT(self, "Failed to create a VapourSynth script environment");
        destroy_graph(self);
        return FALSE;
    }
    self->core = self->vssapi->getCore(self->script);

    VSVideoFormat format;
    if (!self->core || !self->vsapi->getVideoFormatByID(&format, pfYUV420P8, self->core)) {
        GST_ERROR_OBJECT(self, "VapourSynth YUV420P8 format is unavailable");
        destroy_graph(self);
        return FALSE;
    }

    VSVideoInfo input_vi = {
        .format = format,
        .fpsNum = GST_VIDEO_INFO_FPS_N(&self->input_info),
        .fpsDen = GST_VIDEO_INFO_FPS_D(&self->input_info),
        .width = GST_VIDEO_INFO_WIDTH(&self->input_info),
        .height = GST_VIDEO_INFO_HEIGHT(&self->input_info),
        .numFrames = G_MAXINT / 16,
    };
    /* VapourSynth otherwise sizes its cache against total system memory. A
     * temporarily slower 4K graph can then retain many large frames and grow
     * the WebProcess into double-digit GB. Keep enough temporal history for
     * SVP while making memory a bounded cache rather than a latency queue. */
    gint64 frame_bytes = (gint64)input_vi.width * input_vi.height * 3 / 2;
    gint64 cache_bytes = CLAMP(frame_bytes * 64, MIN_CACHE_BYTES, MAX_CACHE_BYTES);
    self->vsapi->setMaxCacheSize(cache_bytes, self->core);
    self->input_node = self->vsapi->createVideoFilter2(
        "LocalBooruInput", &input_vi, input_get_frame, input_free,
        fmParallel, NULL, 0, self, self->core);
    if (!self->input_node) {
        GST_ERROR_OBJECT(self, "Failed to create the VapourSynth input node");
        destroy_graph(self);
        return FALSE;
    }

    VSMap *variables = self->vsapi->createMap();
    self->vsapi->mapSetNode(variables, "video_in", self->input_node, maReplace);
    self->vsapi->mapSetInt(variables, "video_in_dw", input_vi.width, maReplace);
    self->vsapi->mapSetInt(variables, "video_in_dh", input_vi.height, maReplace);
    self->vsapi->mapSetFloat(variables, "container_fps",
                             (double)input_vi.fpsNum / input_vi.fpsDen, maReplace);
    self->vsapi->mapSetFloat(variables, "display_fps", 0.0, maReplace);
    gint64 display_res[2] = { input_vi.width, input_vi.height };
    self->vsapi->mapSetIntArray(variables, "display_res", display_res, 2);
    self->vsapi->mapSetData(variables, "user_data", "", -1, dtUtf8, maReplace);
    self->vssapi->setVariables(self->script, variables);
    self->vsapi->freeMap(variables);

    gchar *script_path = get_script_path();
    if (!script_path || !*script_path) {
        GST_ERROR_OBJECT(self, "No SVP Manager graph is active");
        g_free(script_path);
        destroy_graph(self);
        return FALSE;
    }
    int result = self->vssapi->evaluateFile(self->script, script_path);
    g_free(script_path);
    if (result) {
        const char *error = self->vssapi->getError(self->script);
        GST_ERROR_OBJECT(self, "Failed to evaluate VapourSynth script: %s",
                         error ? error : "unknown error");
        destroy_graph(self);
        return FALSE;
    }

    self->output_node = self->vssapi->getOutputNode(self->script, 0);
    const VSVideoInfo *output_vi = self->output_node
        ? self->vsapi->getVideoInfo(self->output_node)
        : NULL;
    if (!output_vi || output_vi->format.colorFamily != cfYUV ||
        output_vi->format.sampleType != stInteger ||
        output_vi->format.bitsPerSample != 8 ||
        output_vi->format.subSamplingW != 1 || output_vi->format.subSamplingH != 1 ||
        output_vi->fpsNum <= 0 || output_vi->fpsDen <= 0 ||
        output_vi->fpsNum > G_MAXINT || output_vi->fpsDen > G_MAXINT) {
        GST_ERROR_OBJECT(self, "VapourSynth output must be constant-format YUV420P8 video");
        destroy_graph(self);
        return FALSE;
    }

    gint fps_num = (gint)output_vi->fpsNum;
    gint fps_den = (gint)output_vi->fpsDen;
    gst_video_info_set_format(&self->output_info, GST_VIDEO_FORMAT_I420,
                              output_vi->width, output_vi->height);
    GST_VIDEO_INFO_FPS_N(&self->output_info) = fps_num;
    GST_VIDEO_INFO_FPS_D(&self->output_info) = fps_den;
    self->output_duration = gst_util_uint64_scale_int(GST_SECOND, fps_den, fps_num);

    *output_caps = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, "I420",
        "width", G_TYPE_INT, output_vi->width,
        "height", G_TYPE_INT, output_vi->height,
        "framerate", GST_TYPE_FRACTION, fps_num, fps_den,
        NULL);
    GST_INFO_OBJECT(self,
                    "VapourSynth graph initialized: %dx%d %" G_GINT64_FORMAT "/%" G_GINT64_FORMAT
                    " fps -> %dx%d %d/%d fps",
                    input_vi.width, input_vi.height, input_vi.fpsNum, input_vi.fpsDen,
                    output_vi->width, output_vi->height, fps_num, fps_den);

    g_mutex_lock(&self->lock);
    self->shutdown = FALSE;
    self->eof = FALSE;
    self->downstream_flow = GST_FLOW_OK;
    self->worker = g_thread_new("localbooru-vs-output", output_worker, self);
    g_mutex_unlock(&self->lock);

    GST_INFO_OBJECT(self, "VapourSynth bridge initialized: %dx%d %" G_GINT64_FORMAT
                    "/%" G_GINT64_FORMAT " -> %dx%d %d/%d",
                    input_vi.width, input_vi.height, input_vi.fpsNum, input_vi.fpsDen,
                    output_vi->width, output_vi->height, fps_num, fps_den);
    return TRUE;
}

static GstFlowReturn gst_localbooru_vs_chain(GstPad *pad, GstObject *parent,
                                              GstBuffer *buffer) {
    (void)pad;
    GstLocalBooruVS *self = (GstLocalBooruVS *)parent;
    g_mutex_lock(&self->lock);
    gboolean passthrough = self->passthrough;
    g_mutex_unlock(&self->lock);
    if (passthrough)
        return gst_pad_push(self->src_pad, buffer);

    InputFrame *input = g_new0(InputFrame, 1);
    input->buffer = buffer;
    input->pts = GST_BUFFER_PTS(buffer);
    input->duration = GST_BUFFER_DURATION(buffer);

    g_mutex_lock(&self->lock);
    while (!self->shutdown && g_queue_get_length(&self->inputs) >= self->max_buffered)
        g_cond_wait(&self->input_ready, &self->lock);

    if (self->shutdown || self->downstream_flow != GST_FLOW_OK) {
        GstFlowReturn flow = self->downstream_flow != GST_FLOW_OK
            ? self->downstream_flow : GST_FLOW_FLUSHING;
        g_mutex_unlock(&self->lock);
        input_frame_free(input);
        return flow;
    }

    if (!GST_CLOCK_TIME_IS_VALID(self->first_pts) && GST_CLOCK_TIME_IS_VALID(input->pts))
        self->first_pts = input->pts;
    g_queue_push_tail(&self->inputs, input);
    self->input_next++;
    g_cond_broadcast(&self->input_ready);
    g_mutex_unlock(&self->lock);
    return GST_FLOW_OK;
}

static gboolean gst_localbooru_vs_sink_event(GstPad *pad, GstObject *parent,
                                              GstEvent *event) {
    (void)pad;
    GstLocalBooruVS *self = (GstLocalBooruVS *)parent;

    switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS: {
        GstCaps *caps = NULL;
        gst_event_parse_caps(event, &caps);
        if (!gst_video_info_from_caps(&self->input_info, caps) ||
            GST_VIDEO_INFO_FORMAT(&self->input_info) != GST_VIDEO_FORMAT_I420) {
            GST_ELEMENT_ERROR(self, STREAM, FORMAT,
                              ("LocalBooruVS requires I420 input"), (NULL));
            gst_event_unref(event);
            return FALSE;
        }
        gst_caps_replace(&self->input_caps, caps);
        GstCaps *out_caps = NULL;
        gboolean ok = create_graph(self, &out_caps);
        if (!ok) {
            GST_WARNING_OBJECT(self, "VapourSynth graph unavailable; using one unfiltered passthrough pipeline");
            g_mutex_lock(&self->lock);
            self->passthrough = TRUE;
            g_mutex_unlock(&self->lock);
            out_caps = gst_caps_copy(caps);
        } else {
            g_mutex_lock(&self->lock);
            self->passthrough = FALSE;
            g_mutex_unlock(&self->lock);
        }
        gst_event_unref(event);
        ok = gst_pad_push_event(self->src_pad, gst_event_new_caps(out_caps));
        gst_caps_unref(out_caps);
        return ok;
    }
    case GST_EVENT_EOS: {
        g_mutex_lock(&self->lock);
        gboolean passthrough = self->passthrough;
        if (passthrough) {
            g_mutex_unlock(&self->lock);
            return gst_pad_push_event(self->src_pad, event);
        }
        self->eof = TRUE;
        g_cond_broadcast(&self->input_ready);
        g_mutex_unlock(&self->lock);
        gst_event_unref(event);
        return TRUE;
    }
    case GST_EVENT_FLUSH_START:
        destroy_graph(self);
        return gst_pad_push_event(self->src_pad, event);
    case GST_EVENT_FLUSH_STOP: {
        gboolean ok = gst_pad_push_event(self->src_pad, event);
        if (ok && self->input_caps) {
            GstCaps *out_caps = NULL;
            ok = create_graph(self, &out_caps);
            if (!ok) {
                g_mutex_lock(&self->lock);
                self->passthrough = TRUE;
                g_mutex_unlock(&self->lock);
                out_caps = gst_caps_copy(self->input_caps);
            }
            ok = gst_pad_push_event(self->src_pad, gst_event_new_caps(out_caps));
            gst_caps_unref(out_caps);
        }
        return ok;
    }
    case GST_EVENT_SEGMENT:
        g_mutex_lock(&self->lock);
        gboolean has_frames = !g_queue_is_empty(&self->inputs);
        g_mutex_unlock(&self->lock);
        if (has_frames && self->input_caps) {
            GstCaps *out_caps = NULL;
            if (!create_graph(self, &out_caps)) {
                g_mutex_lock(&self->lock);
                self->passthrough = TRUE;
                g_mutex_unlock(&self->lock);
                out_caps = gst_caps_copy(self->input_caps);
            }
            gst_pad_push_event(self->src_pad, gst_event_new_caps(out_caps));
            gst_caps_unref(out_caps);
        }
        return gst_pad_push_event(self->src_pad, event);
    default:
        return gst_pad_push_event(self->src_pad, event);
    }
}

static GstStateChangeReturn gst_localbooru_vs_change_state(GstElement *element,
                                                            GstStateChange transition) {
    GstLocalBooruVS *self = (GstLocalBooruVS *)element;
    if (transition == GST_STATE_CHANGE_PAUSED_TO_READY)
        destroy_graph(self);
    return GST_ELEMENT_CLASS(gst_localbooru_vs_parent_class)->change_state(element, transition);
}

static void gst_localbooru_vs_finalize(GObject *object) {
    GstLocalBooruVS *self = (GstLocalBooruVS *)object;
    destroy_graph(self);
    gst_clear_caps(&self->input_caps);
    g_queue_clear(&self->inputs);
    g_cond_clear(&self->input_ready);
    g_cond_clear(&self->output_ready);
    g_mutex_clear(&self->lock);
    G_OBJECT_CLASS(gst_localbooru_vs_parent_class)->finalize(object);
}

static void gst_localbooru_vs_class_init(GstLocalBooruVSClass *klass) {
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstCaps *caps = gst_caps_from_string(
        "video/x-raw,format=(string)I420,width=(int)[2,MAX],height=(int)[2,MAX],"
        "framerate=(fraction)[1/1,MAX]");

    object_class->finalize = gst_localbooru_vs_finalize;
    element_class->change_state = gst_localbooru_vs_change_state;
    gst_element_class_set_static_metadata(
        element_class,
        "LocalBooru buffered VapourSynth bridge",
        "Filter/Effect/Video",
        "Feeds GStreamer frames into an in-process VapourSynth graph",
        "LocalBooru");
    gst_element_class_add_pad_template(
        element_class, gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_element_class_add_pad_template(
        element_class, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    GST_DEBUG_CATEGORY_INIT(localbooru_vs_debug, "localbooruvs", 0,
                            "LocalBooru VapourSynth bridge");
}

static void gst_localbooru_vs_init(GstLocalBooruVS *self) {
    GstElementClass *klass = GST_ELEMENT_GET_CLASS(self);
    GstPadTemplate *sink_template = gst_element_class_get_pad_template(klass, "sink");
    GstPadTemplate *src_template = gst_element_class_get_pad_template(klass, "src");
    self->sink_pad = gst_pad_new_from_template(sink_template, "sink");
    self->src_pad = gst_pad_new_from_template(src_template, "src");
    gst_pad_set_chain_function(self->sink_pad, GST_DEBUG_FUNCPTR(gst_localbooru_vs_chain));
    gst_pad_set_event_function(self->sink_pad, GST_DEBUG_FUNCPTR(gst_localbooru_vs_sink_event));
    gst_element_add_pad(GST_ELEMENT(self), self->sink_pad);
    gst_element_add_pad(GST_ELEMENT(self), self->src_pad);

    g_mutex_init(&self->lock);
    g_cond_init(&self->input_ready);
    g_cond_init(&self->output_ready);
    g_queue_init(&self->inputs);
    self->max_buffered = DEFAULT_MAX_BUFFERED;
    self->first_pts = GST_CLOCK_TIME_NONE;
    self->first_output = TRUE;
    self->downstream_flow = GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "localbooruvs", GST_RANK_NONE,
                                gst_localbooru_vs_get_type());
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    localbooruvs,
    "LocalBooru buffered VapourSynth bridge",
    plugin_init,
    "0.1.0",
    "LGPL",
    "LocalBooru",
    "https://github.com/DonutsDelivery/LocalBooru")
