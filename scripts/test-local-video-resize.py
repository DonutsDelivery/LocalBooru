#!/usr/bin/env python3
"""Exercise the exact patched scaler using synthetic raw video, never user media."""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("isolation", REPO / "scripts/test-webkit-mpv-isolation.py")
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)

TEST = r'''
struct Expected { int width, height, parN, parD; unsigned count { 0 }; };
static void frame(GstElement*, GstBuffer*, GstPad* pad, gpointer data) {
    auto* expected = static_cast<Expected*>(data);
    GstCaps* caps = gst_pad_get_current_caps(pad); GstVideoInfo info;
    assert(caps && gst_video_info_from_caps(&info, caps)); gst_caps_unref(caps);
    assert(GST_VIDEO_INFO_WIDTH(&info) == expected->width);
    assert(GST_VIDEO_INFO_HEIGHT(&info) == expected->height);
    assert(GST_VIDEO_INFO_PAR_N(&info) == expected->parN);
    assert(GST_VIDEO_INFO_PAR_D(&info) == expected->parD);
    assert(GST_VIDEO_INFO_FPS_N(&info) == 24 && GST_VIDEO_INFO_FPS_D(&info) == 1);
    assert(GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_I420);
    ++expected->count;
}
static void run(int width, int height, int targetW, int targetH, int expectedW, int expectedH,
    bool graph, const std::string& geometry, int parN = 1, int parD = 1) {
    GstElement* pipeline = gst_pipeline_new(nullptr);
    GstElement* source = gst_element_factory_make("videotestsrc", nullptr);
    GstElement* input = gst_element_factory_make("capsfilter", nullptr);
    GstElement* sink = gst_element_factory_make("fakesink", nullptr);
    assert(pipeline && source && input && sink);
    GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "I420",
        "width", G_TYPE_INT, width, "height", G_TYPE_INT, height,
        "framerate", GST_TYPE_FRACTION, 24, 1, "pixel-aspect-ratio", GST_TYPE_FRACTION, parN, parD, nullptr);
    g_object_set(input, "caps", caps, nullptr); gst_caps_unref(caps);
    g_object_set(source, "num-buffers", 4, nullptr);
    g_object_set(sink, "sync", false, "signal-handoffs", true, nullptr);
    Expected expected { expectedW, expectedH, parN, parD };
    g_signal_connect(sink, "handoff", G_CALLBACK(frame), &expected);
    GstElement* opaqueGraph = graph ? gst_element_factory_make("identity", "unchanged-manager-graph") : nullptr;
    LocalBooruResizeContract contract { 1, 3, static_cast<guint>(targetW), static_cast<guint>(targetH), geometry };
    GstElement* filter = localBooruComposeVideoFilter(targetW ? &contract : nullptr, opaqueGraph);
    // Observe the graph input separately: resizing must occur before optional SVP.
    Expected upstream { expectedW, expectedH, parN, parD };
    if (opaqueGraph) {
        GstPad* pad = gst_element_get_static_pad(opaqueGraph, "sink");
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, [](GstPad* pad, GstPadProbeInfo*, gpointer data) {
            frame(nullptr, nullptr, pad, data); return GST_PAD_PROBE_OK;
        }, &upstream, nullptr); gst_object_unref(pad);
    }
    gst_bin_add_many(GST_BIN(pipeline), source, input, sink, nullptr);
    if (filter) { gst_bin_add(GST_BIN(pipeline), filter); assert(gst_element_link_many(source, input, filter, sink, nullptr)); }
    else assert(gst_element_link_many(source, input, sink, nullptr));
    assert(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE);
    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* message = gst_bus_timed_pop_filtered(bus, 10 * GST_SECOND, static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
    if (message && GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* error = nullptr; gchar* detail = nullptr; gst_message_parse_error(message, &error, &detail);
        std::cerr << error->message << " " << (detail ? detail : "") << "\n";
    }
    assert(message && GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS);
    gst_message_unref(message); gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    assert(expected.count == 4); if (graph) assert(upstream.count == 4);
    if (targetW) {
        gchar* contents = nullptr; assert(g_file_get_contents(geometry.c_str(), &contents, nullptr, nullptr));
        std::string expectedGeometry = "1\n3\n" + std::to_string(width) + "\n" + std::to_string(height) + "\n"
            + std::to_string(expectedW) + "\n" + std::to_string(expectedH) + "\n";
        assert(std::string(contents) == expectedGeometry); g_free(contents);
    }
    gst_object_unref(pipeline);
}
int main(int argc, char** argv) {
    assert(argc == 2); gst_init(&argc, &argv); std::string root = argv[1];
    g_setenv("LOCALBOORU_MPV_CONTROL_HOST_ROOT", root.c_str(), true);
    const char* id = "localbooru-svp-host-synthetic";
    std::string path = root + "/" + id;
    auto put = [](const std::string& path, const std::string& text) { assert(g_file_set_contents(path.c_str(), text.c_str(), text.size(), nullptr)); };
    put(root + "/resize-active", std::string(id) + "\n1\n3\n");
    put(path + ".resize", "1\n3\n1280\n720\n");
    LocalBooruResizeContract contract;
    assert(localBooruReadResizeContract(id, contract)); assert(contract.width == 1280 && contract.height == 720);
    // AC: @local-decoded-video-resolution ac-ownership
    put(path + ".resize", "1\n2\n1280\n720\n"); assert(!localBooruReadResizeContract(id, contract));
    for (const char* bad : { "1\n3\n0\n720\n", "1\n3\n16385\n720\n", "1\n3\n-1\n720\n", "1\n3\n1280\n720\nextra", "0\n3\n1280\n720\n" }) {
        put(path + ".resize", bad); assert(!localBooruReadResizeContract(id, contract));
    }
    unlink((path + ".resize").c_str()); assert(!symlink((root + "/resize-active").c_str(), (path + ".resize").c_str()));
    assert(!localBooruReadResizeContract(id, contract)); unlink((path + ".resize").c_str());
    assert(!localBooruReadResizeContract("../invalid", contract));
    // AC: @local-decoded-video-resolution ac-local-raw
    // AC: @local-decoded-video-resolution ac-geometry
    for (bool graph : { false, true }) {
        run(3840, 2160, 1280, 720, 1280, 720, graph, path + ".geometry");
        run(2160, 3840, 1280, 720, 404, 720, graph, path + ".geometry");
        run(1920, 1080, 854, 480, 852, 480, graph, path + ".geometry");
        run(640, 360, 1280, 720, 640, 360, graph, path + ".geometry");
        run(1920, 1080, 1280, 720, 1280, 720, graph, path + ".geometry", 4, 3);
        run(641, 361, 0, 0, 641, 361, graph, path + ".geometry");
    }
    std::cout << "Raw caps negotiation passed: landscape, portrait, even bounds, no upscale, PAR/FPS, Original, pre-graph geometry\n";
}
'''


def main(source):
    with tempfile.TemporaryDirectory(prefix="dmc-synthetic-raw-resize-") as directory:
        root = Path(directory)
        program = root / "source"
        for line in isolation.prepare.PATCH.read_text().splitlines():
            if line.startswith("# preimage "):
                relative = line.split()[-1]
                destination = program / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, destination)
        print(isolation.prepare.prepare(program, apply=True))
        media = (program / "Source/WebCore/platform/graphics/gstreamer/MediaPlayerPrivateGStreamer.cpp").read_text()
        helpers = "// DMC's selected video element" + media.split("// DMC's selected video element", 1)[1].split("WTF_MAKE_TZONE_ALLOCATED_IMPL", 1)[0]
        cpp = root / "fixture.cpp"
        cpp.write_text(isolation.HEADER + helpers + TEST)
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0", "gstreamer-video-1.0"], text=True))
        helper = os.environ.get("HOST_HEAVY_BUILD_HELPER", str(Path.home() / ".local/bin/host-heavy-build"))
        subprocess.run([helper, "run", "--project", "localbooru-raw-resize-test", "--worktree", str(REPO), "--wait", "0", "--", "g++", "-std=c++17", "-pthread", str(cpp), "-o", str(root / "fixture"), *flags], check=True)
        subprocess.run([str(root / "fixture"), str(root)], check=True, timeout=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    main(parser.parse_args().source)
