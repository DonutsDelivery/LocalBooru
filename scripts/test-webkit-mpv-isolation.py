#!/usr/bin/env python3
"""Run patched C++ helpers against disposable sockets, never the user's sockets."""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare", REPO / "scripts/prepare-webkit-mpv-isolation.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

HEADER = r'''
#include <glib.h>
#include <gst/video/video.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
struct GFree { void operator()(char* data) const { g_free(data); } };
template<class T> using GUniquePtr = std::unique_ptr<T, GFree>;
'''

TEST = r'''
static void put(const std::string& path, const std::string& value) { std::ofstream(path) << value; }
static int socketAt(const std::string& path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0); assert(fd >= 0);
    sockaddr_un address { }; address.sun_family = AF_UNIX;
    assert(path.size() < sizeof(address.sun_path)); strcpy(address.sun_path, path.c_str());
    assert(!bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address))); assert(!listen(fd, 8)); return fd;
}
template<class F> static bool waitFor(F predicate) {
    for (int n = 0; n < 300; ++n) {
        while (g_main_context_iteration(nullptr, false)) { }
        if (predicate()) return true;
        g_usleep(10000);
    }
    return false;
}
int main(int argc, char** argv) {
    assert(argc == 2); std::string root = argv[1];
    assert(!mkdir((root + "/hosts").c_str(), 0700));
    std::string hosts = root + "/hosts";
    g_setenv("LOCALBOORU_MPV_CONTROL_HOST_ROOT", hosts.c_str(), true);
    const char* id = "localbooru-svp-host-video"; std::string marker = hosts + "/" + id;
    int player = 1, other = 2;
    // AC: @svp-platform-routing ac-linux-route
    localBooruRegisterVideoHost(&player, id); assert(!access(marker.c_str(), F_OK));
    std::string registered; assert(localBooruPrivateHostFile(marker, registered));
    assert(registered == std::to_string(getpid()) + "\n");
    localBooruRegisterVideoHost(&other, id); // Duplicate ID cannot steal ownership.
    assert(localBooruVideoHosts.count(&other) == 0);
    localBooruRegisterVideoHost(&other, "../outside"); assert(!access(marker.c_str(), F_OK));
    std::string active = std::string(id) + "\n" + std::to_string(getpid()) + "\n1\n2\n";
    put(hosts + "/active", active); assert(localBooruIsActiveVideoHost(hosts.c_str()));
    for (const auto& malformed : { std::string(id) + "\n999999\n1\n2\n", active + "extra", std::string(id) + "\n" + std::to_string(getpid()) + "\n0\n2\n", std::string(id) + "\n" + std::to_string(getpid()) + "\n1\n" }) {
        put(hosts + "/active", malformed); assert(!localBooruIsActiveVideoHost(hosts.c_str()));
    }
    put(hosts + "/replacement", "999999\n"); assert(!rename((hosts + "/replacement").c_str(), marker.c_str()));
    localBooruUnregisterVideoHost(&player); // Old player cannot delete replacement inode.
    assert(!access(marker.c_str(), F_OK));
    localBooruRegisterVideoHost(&other, id); assert(localBooruVideoHosts.count(&other) == 0);
    unlink(marker.c_str()); localBooruRegisterVideoHost(&player, id);
    unlink((hosts + "/active").c_str());
    std::string defaultSocket = root + "/mpvsocket", peerSocket = root + "/mpvSockets/424242";
    assert(!mkdir((root + "/mpvSockets").c_str(), 0700));
    int defaultFD = socketAt(defaultSocket), peerFD = socketAt(peerSocket);
    std::string upstream = root + "/upstream"; int upstreamFD = socketAt(upstream);
    std::thread server([&] { int fd = accept(upstreamFD, nullptr, nullptr); assert(fd >= 0); char buffer[4]; assert(read(fd, buffer, 4) == 4); assert(write(fd, buffer, 4) == 4); close(fd); });
    g_thread_unref(g_thread_new("synthetic relay", localBooruMpvControlRelay, g_strdup(upstream.c_str())));
    std::string published = root + "/mpvSockets/" + std::to_string(getpid());
    g_usleep(300000); assert(access(published.c_str(), F_OK)); // Idle stays private.
    put(hosts + "/active", active);
    assert(waitFor([&] { return !access(published.c_str(), F_OK); }));
    int client = localBooruConnectUnixSocket(published.c_str()); assert(client >= 0);
    assert(write(client, "test", 4) == 4); char response[4]; assert(read(client, response, 4) == 4); assert(!memcmp(response, "test", 4)); close(client); server.join();
    assert(!access(defaultSocket.c_str(), F_OK)); assert(!access(peerSocket.c_str(), F_OK));
    unlink((hosts + "/active").c_str());
    assert(waitFor([&] { return access(published.c_str(), F_OK); }));
    assert(!access(defaultSocket.c_str(), F_OK)); assert(!access(peerSocket.c_str(), F_OK));
    put(hosts + "/active", active); assert(waitFor([&] { return !access(published.c_str(), F_OK); }));
    unlink(published.c_str()); int successor = socketAt(published);
    unlink((hosts + "/active").c_str()); g_usleep(400000);
    assert(!access(published.c_str(), F_OK)); // Cleanup preserves replacement socket.
    localBooruUnregisterVideoHost(&player); assert(access(marker.c_str(), F_OK));
    close(successor); close(upstreamFD); close(defaultFD); close(peerFD);
    std::cout << "C++ registration/readiness/relay/coexistence/cleanup passed\n";
}
'''


def main(source):
    assert prepare.validate_plan("ninja: no work to do.\n") == []
    allowed = "\n".join(f"[{n}/3] Building CXX object {path}" for n, path in enumerate(prepare.OBJECTS, 1))
    allowed += "\n[3/3] Linking CXX shared library lib/libwebkit2gtk-4.1.so.0.21.7"
    assert len(prepare.validate_plan(allowed)) == 3
    for unsafe in ["[1/533] Generate bindings (WebCoreBindings)",
                   "BUILD Generate bindings\nBUILD Building CXX object unexpected"]:
        try:
            prepare.validate_plan(unsafe)
            raise AssertionError("unrelated cache work accepted")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory(prefix="dmc-synthetic-webkit-isolation-") as directory:
        root = Path(directory)
        program = root / "source"
        entries = [line.split()[-1] for line in prepare.PATCH.read_text().splitlines()
                   if line.startswith("# preimage ")]
        for relative in entries:
            target = program / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        print(prepare.prepare(program, apply=True))
        assert prepare.prepare(program) == "already upgraded"
        # An unrelated edit must be rejected without altering source.
        check = program / entries[0]
        exact = check.read_bytes()
        check.write_bytes(exact + b"\n// synthetic unknown preimage\n")
        try:
            prepare.prepare(program, apply=True)
            raise AssertionError("unknown preimage accepted")
        except ValueError:
            pass
        check.write_bytes(exact)
        process = check.read_text()
        relay = process.split("namespace {", 1)[1].split("} // namespace", 1)[0]
        # Redirect ONLY the synthetic compiled fixture. Production uses the
        # documented mpvSockets PID directory; tests never touch that directory.
        relay = relay.replace('"/tmp/mpvSockets"', '"' + str(root / "mpvSockets") + '"')
        media = (program / entries[1]).read_text()
        registration = "// DMC's selected video element" + media.split("// DMC's selected video element", 1)[1].split("WTF_MAKE_TZONE_ALLOCATED_IMPL", 1)[0]
        cpp = root / "fixture.cpp"
        cpp.write_text(HEADER + relay + registration + TEST)
        flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "glib-2.0", "gstreamer-video-1.0"], text=True))
        helper = os.environ.get("HOST_HEAVY_BUILD_HELPER", str(Path.home() / ".local/bin/host-heavy-build"))
        subprocess.run([helper, "run", "--project", "localbooru-svp-isolation-test", "--worktree", str(REPO), "--wait", "0", "--", "g++", "-std=c++17", "-pthread", str(cpp), "-o", str(root / "fixture"), *flags], check=True)
        subprocess.run([str(root / "fixture"), str(root)], check=True, timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    main(parser.parse_args().source)
