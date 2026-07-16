#define _GNU_SOURCE

#include <dlfcn.h>
#include <gst/gst.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef GstElement *(*GstElementFactoryMakeFn)(const gchar *, const gchar *);

static GstElementFactoryMakeFn real_factory_make(void) {
    static GstElementFactoryMakeFn function = NULL;
    if (!function) {
        function = (GstElementFactoryMakeFn)dlsym(RTLD_NEXT, "gst_element_factory_make");
    }
    return function;
}

GstElement *gst_element_factory_make(const gchar *factory_name, const gchar *name) {
    GstElementFactoryMakeFn make = real_factory_make();
    if (!make) {
        return NULL;
    }

    GstElement *element = make(factory_name, name);
    if (!element || !factory_name || (strcmp(factory_name, "playbin") != 0 && strcmp(factory_name, "playbin3") != 0)) {
        return element;
    }

    const char *filter_factory = getenv("WEBKIT_GST_VIDEO_FILTER");
    if (!filter_factory || !*filter_factory) {
        return element;
    }

    GstElement *filter = make(filter_factory, NULL);
    if (!filter) {
        fprintf(stderr, "[WebKitGstFilterHook] unable to create video filter '%s'\n", filter_factory);
        return element;
    }

    g_object_set(element, "video-filter", filter, NULL);
    fprintf(stderr, "[WebKitGstFilterHook] attached '%s' to %s\n", filter_factory, factory_name);
    gst_object_unref(filter);
    return element;
}
