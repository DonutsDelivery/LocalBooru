#include <gst/gst.h>

#ifndef PACKAGE
#define PACKAGE "localbooru-gst-filter"
#endif

typedef struct {
    GstBin parent;
} GstLocalBooruScale720;

typedef struct {
    GstBinClass parent_class;
} GstLocalBooruScale720Class;

typedef struct {
    GstBin parent;
} GstLocalBooruScale1080;

typedef struct {
    GstBinClass parent_class;
} GstLocalBooruScale1080Class;

G_DEFINE_TYPE(GstLocalBooruScale720, gst_localbooru_scale_720, GST_TYPE_BIN)
G_DEFINE_TYPE(GstLocalBooruScale1080, gst_localbooru_scale_1080, GST_TYPE_BIN)

static GstPadProbeReturn report_output_caps(
    GstPad *pad,
    GstPadProbeInfo *info,
    gpointer user_data) {
    (void)pad;
    const gchar *quality = user_data;
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
        GstCaps *caps = NULL;
        gst_event_parse_caps(event, &caps);
        gchar *description = gst_caps_to_string(caps);
        g_message("[LocalBooruScale] %s negotiated output %s", quality, description);
        g_free(description);
        return GST_PAD_PROBE_REMOVE;
    }
    return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn report_first_output_buffer(
    GstPad *pad,
    GstPadProbeInfo *info,
    gpointer user_data) {
    (void)pad;
    const gchar *quality = user_data;
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    const guint memory_count = gst_buffer_n_memory(buffer);
    const gchar *memory_type = "none";
    if (memory_count > 0) {
        GstMemory *memory = gst_buffer_peek_memory(buffer, 0);
        if (memory->allocator != NULL && memory->allocator->mem_type != NULL)
            memory_type = memory->allocator->mem_type;
    }
    g_message(
        "[LocalBooruScale] %s first output buffer size=%" G_GSIZE_FORMAT
        " memories=%u allocator=%s",
        quality,
        gst_buffer_get_size(buffer),
        memory_count,
        memory_type);
    return GST_PAD_PROBE_REMOVE;
}

static void configure_scale_bin(GstBin *bin, gint output_height, const gchar *quality) {
    GstElement *scale = gst_element_factory_make("videoscale", NULL);
    GstElement *upload = gst_element_factory_make("glupload", NULL);
    GstElement *convert = gst_element_factory_make("glcolorconvert", NULL);
    GstElement *caps_filter = gst_element_factory_make("capsfilter", NULL);
    const gint output_width = output_height == 720 ? 1280 : 1920;
    GstCaps *caps = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, "NV12",
        "width", G_TYPE_INT, output_width,
        "height", G_TYPE_INT, output_height,
        "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1,
        NULL);
    gst_caps_set_features(
        caps,
        0,
        gst_caps_features_new("memory:GLMemory", NULL));

    g_return_if_fail(scale != NULL);
    g_return_if_fail(upload != NULL);
    g_return_if_fail(convert != NULL);
    g_return_if_fail(caps_filter != NULL);
    g_return_if_fail(caps != NULL);

    g_object_set(scale, "method", 1, NULL);
    g_object_set(caps_filter, "caps", caps, NULL);
    gst_caps_unref(caps);

    gst_bin_add_many(bin, scale, upload, convert, caps_filter, NULL);
    g_return_if_fail(gst_element_link_many(scale, upload, convert, caps_filter, NULL));

    GstPad *sink = gst_element_get_static_pad(scale, "sink");
    GstPad *src = gst_element_get_static_pad(caps_filter, "src");
    g_return_if_fail(sink != NULL);
    g_return_if_fail(src != NULL);
    g_return_if_fail(gst_element_add_pad(GST_ELEMENT(bin), gst_ghost_pad_new("sink", sink)));
    g_return_if_fail(gst_element_add_pad(GST_ELEMENT(bin), gst_ghost_pad_new("src", src)));
    gst_pad_add_probe(
        src,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        report_output_caps,
        (gpointer)quality,
        NULL);
    gst_pad_add_probe(
        src,
        GST_PAD_PROBE_TYPE_BUFFER,
        report_first_output_buffer,
        (gpointer)quality,
        NULL);
    gst_object_unref(sink);
    gst_object_unref(src);
}

static void gst_localbooru_scale_720_class_init(GstLocalBooruScale720Class *klass) {
    gst_element_class_set_static_metadata(
        GST_ELEMENT_CLASS(klass),
        "LocalBooru 720p playback scaler",
        "Filter/Video",
        "Scales decoded playback frames to 720p without encoding",
        "LocalBooru");
}

static void gst_localbooru_scale_720_init(GstLocalBooruScale720 *self) {
    configure_scale_bin(GST_BIN(self), 720, "720p");
}

static void gst_localbooru_scale_1080_class_init(GstLocalBooruScale1080Class *klass) {
    gst_element_class_set_static_metadata(
        GST_ELEMENT_CLASS(klass),
        "LocalBooru 1080p playback scaler",
        "Filter/Video",
        "Scales decoded playback frames to 1080p without encoding",
        "LocalBooru");
}

static void gst_localbooru_scale_1080_init(GstLocalBooruScale1080 *self) {
    configure_scale_bin(GST_BIN(self), 1080, "1080p");
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(
               plugin,
               "localbooruscale720",
               GST_RANK_NONE,
               gst_localbooru_scale_720_get_type())
        && gst_element_register(
               plugin,
               "localbooruscale1080",
               GST_RANK_NONE,
               gst_localbooru_scale_1080_get_type());
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    localbooruscale,
    "LocalBooru decoded-frame playback scalers",
    plugin_init,
    "0.1.0",
    "MIT",
    "LocalBooru",
    "https://github.com/DonutsDelivery/LocalBooru")
