// SPDX-License-Identifier: MIT
// Installed as /usr/bin/localbooru by the Linux package builder. The real Tauri
// executable lives under /usr/lib/localbooru so this launcher can select the
// bundled patched WebKit runtime before the dynamic loader starts the app.
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int prepend_env(const char *name, const char *value) {
    const char *current = getenv(name);
    size_t length = strlen(value) + (current && *current ? strlen(current) + 1 : 0) + 1;
    char *combined = malloc(length);
    if (!combined)
        return -1;
    if (current && *current)
        snprintf(combined, length, "%s:%s", value, current);
    else
        snprintf(combined, length, "%s", value);
    int result = setenv(name, combined, 1);
    free(combined);
    return result;
}

int main(int argc, char **argv) {
    (void)argc;
    const char *root = "/usr/lib/localbooru";
    const char *runtime = "/usr/lib/localbooru/native-svp";
    char path[PATH_MAX];

    snprintf(path, sizeof(path), "%s/lib", runtime);
    if (prepend_env("LD_LIBRARY_PATH", path) != 0)
        goto env_error;
    snprintf(path, sizeof(path), "%s/python-home/lib", runtime);
    if (prepend_env("LD_LIBRARY_PATH", path) != 0)
        goto env_error;
    snprintf(path, sizeof(path), "%s/python-home/lib/python3.12/site-packages", runtime);
    if (prepend_env("PYTHONPATH", path) != 0)
        goto env_error;
    snprintf(path, sizeof(path), "%s/python-home", runtime);
    if (setenv("PYTHONHOME", path, 1) != 0)
        goto env_error;
    snprintf(path, sizeof(path), "%s/gstreamer", runtime);
    if (prepend_env("GST_PLUGIN_PATH", path) != 0)
        goto env_error;
    if (setenv("LOCALBOORU_GSTREAMER_SVP_DIR", path, 1) != 0)
        goto env_error;
    snprintf(path, sizeof(path), "%s/bin/mpv", runtime);
    if (setenv("LOCALBOORU_WEB_PROCESS_PATH", path, 1) != 0)
        goto env_error;
    if (!getenv("LOCALBOORU_ENABLE_NATIVE_SVP") &&
        setenv("LOCALBOORU_ENABLE_NATIVE_SVP", "1", 1) != 0)
        goto env_error;

    snprintf(path, sizeof(path), "%s/localbooru", root);
    argv[0] = path;
    execv(path, argv);
    fprintf(stderr, "LocalBooru launcher: exec %s failed: %s\n", path, strerror(errno));
    return 127;

env_error:
    fprintf(stderr, "LocalBooru launcher: failed to configure runtime: %s\n", strerror(errno));
    return 126;
}
