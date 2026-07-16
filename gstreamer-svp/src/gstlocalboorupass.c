#include <gst/gst.h>
#include <gst/video/gstvideofilter.h>

#ifndef PACKAGE
#define PACKAGE "localbooru-gst-filter"
#endif

typedef struct {
    GstVideoFilter parent;
} GstLocalBooruPass;

typedef struct {
    GstVideoFilterClass parent_class;
} GstLocalBooruPassClass;

G_DEFINE_TYPE(GstLocalBooruPass, gst_localbooru_pass, GST_TYPE_VIDEO_FILTER)

static GstFlowReturn gst_localbooru_pass_transform_frame_ip(GstVideoFilter *filter, GstVideoFrame *frame) {
    (void)filter;
    (void)frame;
    return GST_FLOW_OK;
}

static void gst_localbooru_pass_class_init(GstLocalBooruPassClass *klass) {
    GstElementClass *element_class = GST_ELEMENT_CLASS(klass);
    GstVideoFilterClass *filter_class = GST_VIDEO_FILTER_CLASS(klass);
    GstCaps *caps = gst_caps_from_string("video/x-raw(ANY)");

    gst_element_class_set_static_metadata(
        element_class,
        "LocalBooru WebKit passthrough filter",
        "Filter/Video",
        "Passthrough used to validate WebKit playbin filter attachment",
        "LocalBooru");
    gst_element_class_add_pad_template(
        element_class,
        gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
    gst_element_class_add_pad_template(
        element_class,
        gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    gst_caps_unref(caps);

    filter_class->transform_frame_ip = gst_localbooru_pass_transform_frame_ip;
}

static void gst_localbooru_pass_init(GstLocalBooruPass *self) {
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(plugin, "localboorupass", GST_RANK_NONE, gst_localbooru_pass_get_type());
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    localboorupass,
    "LocalBooru WebKit video-filter attachment probe",
    plugin_init,
    "0.1.0",
    "LGPL",
    "LocalBooru",
    "https://github.com/DonutsDelivery/LocalBooru")
