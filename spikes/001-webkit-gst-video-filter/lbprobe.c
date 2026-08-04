#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>

#ifndef PACKAGE
#define PACKAGE "localbooru-gst-filter-spike"
#endif

typedef struct {
  GstVideoFilter parent;
  guint64 frames;
} GstLbProbe;

typedef struct {
  GstVideoFilterClass parent_class;
} GstLbProbeClass;

G_DEFINE_TYPE(GstLbProbe, gst_lb_probe, GST_TYPE_VIDEO_FILTER)

static gboolean gst_lb_probe_set_info(GstVideoFilter *filter,
                                      GstCaps *incaps,
                                      GstVideoInfo *in_info,
                                      GstCaps *outcaps,
                                      GstVideoInfo *out_info) {
  (void)filter;
  (void)incaps;
  (void)outcaps;
  g_printerr("LB_FILTER_NEGOTIATED %ux%u %s -> %s\n",
             GST_VIDEO_INFO_WIDTH(in_info), GST_VIDEO_INFO_HEIGHT(in_info),
             gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(in_info)),
             gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(out_info)));
  return TRUE;
}

static GstFlowReturn gst_lb_probe_transform_frame_ip(GstVideoFilter *filter,
                                                      GstVideoFrame *frame) {
  GstLbProbe *self = (GstLbProbe *)filter;
  self->frames++;
  if (self->frames == 1 || self->frames % 30 == 0) {
    g_printerr("LB_FILTER_FRAME count=%" G_GUINT64_FORMAT " pts=%" GST_TIME_FORMAT "\n",
               self->frames, GST_TIME_ARGS(GST_BUFFER_PTS(frame->buffer)));
  }
  return GST_FLOW_OK;
}

static void gst_lb_probe_class_init(GstLbProbeClass *klass) {
  GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
  GstVideoFilterClass *filter_class = GST_VIDEO_FILTER_CLASS(klass);
  GstCaps *caps = gst_caps_from_string("video/x-raw");

  gst_element_class_set_static_metadata(
      element_class,
      "LocalBooru WebKit video-filter probe",
      "Filter/Video",
      "Pass-through filter proving playbin video-filter insertion",
      "LocalBooru spike");
  gst_element_class_add_pad_template(
      element_class,
      gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template(
      element_class,
      gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);

  filter_class->set_info = gst_lb_probe_set_info;
  filter_class->transform_frame_ip = gst_lb_probe_transform_frame_ip;
}

static void gst_lb_probe_init(GstLbProbe *self) {
  self->frames = 0;
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "lbprobe", GST_RANK_NONE,
                              gst_lb_probe_get_type());
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR,
                  GST_VERSION_MINOR,
                  lbprobe,
                  "LocalBooru playbin video-filter feasibility probe",
                  plugin_init,
                  "0.1.0",
                  "LGPL",
                  "LocalBooru",
                  "https://localbooru.invalid")
