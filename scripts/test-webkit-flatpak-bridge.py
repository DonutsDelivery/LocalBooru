#!/usr/bin/env python3
"""Check the pinned patch and its exact path guard using owned fixtures only."""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bridge", REPO / "scripts/prepare-webkit-flatpak-bridge.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)
RELATIVE = "Source/WebKit/UIProcess/Launcher/glib/FlatpakLauncher.cpp"

HEADER = r'''
#include <glib.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
struct GFree { void operator()(char* data) const { g_free(data); } };
template<class T> using GUniquePtr = std::unique_ptr<T, GFree>;
'''

TEST = r'''
int main(int argc, char** argv) {
    assert(argc == 2);
    g_unsetenv("LOCALBOORU_SVP_SNAPSHOT_ROOT");
    assert(!localBooruFlatpakBridgeRoot());
    auto selected = [](const std::string& path) {
        g_setenv("LOCALBOORU_SVP_SNAPSHOT_ROOT", path.c_str(), true);
        return localBooruFlatpakBridgeRoot() != nullptr;
    };
    std::string base = argv[1];
    std::string root = base + "/localbooru-svp-snapshots-synthetic";
    assert(!mkdir(root.c_str(), 0700));
    assert(selected(root));
    assert(!selected("relative/localbooru-svp-snapshots-synthetic"));
    assert(!selected(base)); // Never expose all app data or another broad root.
    assert(!selected(root + "/../localbooru-svp-snapshots-synthetic"));
    assert(!chmod(root.c_str(), 0755)); assert(!selected(root));
    assert(!chmod(root.c_str(), 0700)); assert(selected(root));
    std::string link = base + "/localbooru-svp-snapshots-link";
    assert(!symlink(root.c_str(), link.c_str()));
    assert(!selected(link)); assert(!selected(link + "/"));
    std::string file = base + "/localbooru-svp-snapshots-file";
    assert(g_file_set_contents(file.c_str(), "fixture", -1, nullptr));
    assert(!selected(file)); assert(!selected(file + "-missing"));
    std::string empty = base + "/localbooru-svp-snapshots-";
    assert(!mkdir(empty.c_str(), 0700)); assert(!selected(empty));
    std::cout << "Private Flatpak bridge guard passed: owned directory, spaces, permissions, canonical paths, symlinks, files and missing paths\n";
}
'''


def main(source, compile_guard):
    with tempfile.TemporaryDirectory(prefix="dmc-flatpak-bridge-æ space-") as directory:
        root = Path(directory).resolve()
        program = root / "source"
        target = program / RELATIVE
        target.parent.mkdir(parents=True)
        shutil.copyfile(source / RELATIVE, target)
        process_relative = "Source/WebKit/Shared/glib/ProcessExecutablePathGLib.cpp"
        process_target = program / process_relative
        process_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / process_relative, process_target)
        process_patch = REPO / "patches/webkitgtk/2.52.3-bundled-process-directory.patch"
        bridge.prepare(program, apply=True, patch=process_patch)
        assert bridge.prepare(program, apply=True, patch=process_patch) == "already upgraded"
        bridge.prepare(program, apply=True)
        assert bridge.prepare(program, apply=True) == "already upgraded"
        patched = target.read_text()
        target.write_text(patched + "\n// unrecognized local edit\n")
        try:
            bridge.prepare(program, apply=True)
            raise AssertionError("Unknown native cache modification was accepted")
        except ValueError:
            pass
        assert target.read_text().endswith("// unrecognized local edit\n")
        print("Pinned patch applies, is idempotent, and refuses unknown cache edits", flush=True)
        if not compile_guard:
            return
        helper = "static const char* localBooruFlatpakBridgeRoot()" + patched.split(
            "static const char* localBooruFlatpakBridgeRoot()", 1)[1].split("GRefPtr<GSubprocess> flatpakSpawn", 1)[0]
        cpp = root / "fixture.cpp"
        process_helper = "static String findWebKitProcess(const char* processName)" + process_target.read_text().split(
            "static String findWebKitProcess(const char* processName)", 1)[1].split("String executablePathOfWebProcess()", 1)[0]
        process_types = r'''
#define ENABLE(feature) 0
#define PKGLIBEXECDIR "/system-webkit-helpers"
using String = std::string;
namespace FileSystem {
static String stringFromFileSystemRepresentation(const char* path) { return path; }
static String pathByAppendingComponent(const String& parent, const String& child) { return parent + "/" + child; }
}
namespace StringView { static String fromLatin1(const char* text) { return text; } }
'''
        process_test = r'''
static void testProcessLookup() {
    g_unsetenv("LOCALBOORU_WEBKIT_PROCESS_DIR");
    assert(findWebKitProcess("WebKitNetworkProcess") == "/system-webkit-helpers/WebKitNetworkProcess");
    g_setenv("LOCALBOORU_WEBKIT_PROCESS_DIR", "/owned fixture with spaces", true);
    // A missing packaged helper must not silently use a different host version.
    assert(findWebKitProcess("WebKitNetworkProcess") == "/owned fixture with spaces/WebKitNetworkProcess");
    assert(findWebKitProcess("WebKitGPUProcess") == "/owned fixture with spaces/WebKitGPUProcess");
    g_unsetenv("LOCALBOORU_WEBKIT_PROCESS_DIR");
}
'''
        cpp.write_text(HEADER + helper + process_types + process_helper + process_test + TEST.replace("assert(argc == 2);", "testProcessLookup(); assert(argc == 2);"))
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0"], text=True))
        gate = os.environ.get("HOST_HEAVY_BUILD_HELPER", str(Path.home() / ".local/bin/host-heavy-build"))
        subprocess.run([gate, "run", "--project", "DonutMediaCenter", "--worktree", str(REPO), "--wait", "21600", "--",
                        "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(root / "fixture"), *flags], check=True)
        subprocess.run([str(root / "fixture"), str(root)], check=True, timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--compile-guard", action="store_true", help="compile the exact guard under the shared host build gate")
    args = parser.parse_args()
    main(args.source, args.compile_guard)
