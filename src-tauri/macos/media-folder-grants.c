/* Linked directly into the sandboxed media helpers; see media-folder-grants.md. */
#include <CoreFoundation/CoreFoundation.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_GRANTS 1024
#define MAX_BOOKMARK (1024 * 1024)
#define MAX_TOTAL (16 * 1024 * 1024)

static CFURLRef scopes[MAX_GRANTS];
static size_t scope_count;

static void cleanup(void) {
    while (scope_count) {
        CFURLRef url = scopes[--scope_count];
        CFURLStopAccessingSecurityScopedResource(url);
        CFRelease(url);
    }
}

static void fail(void) {
    cleanup();
    fputs("Media folder grant initialization failed.\n", stderr);
    _exit(78);
}

static int valid_name(const char *name) {
    size_t n = strlen(name), i;
    if (n <= 9 || strcmp(name + n - 9, ".bookmark") != 0) return 0;
    for (i = 0; i < n - 9; i++) {
        unsigned char c = (unsigned char)name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '-')) return 0;
    }
    return 1;
}

__attribute__((constructor)) static void activate_media_grants(void) {
    const char *value = getenv("DMC_MEDIA_BOOKMARK_DIRECTORY");
    char *path;
    DIR *dir;
    struct dirent *entry;
    size_t count = 0, total = 0;
    int fd;
    if (!value) return;
    path = strdup(value);
    if (unsetenv("DMC_MEDIA_BOOKMARK_DIRECTORY") != 0 || !path) fail();
    /* A trailing slash would otherwise force traversal of a final symlink. */
    { size_t len = strlen(path);
      while (len > 1 && path[len - 1] == '/') path[--len] = '\0'; }
    fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    free(path);
    if (fd < 0) fail();
    dir = fdopendir(fd);
    if (!dir) { close(fd); fail(); }
    for (;;) {
        struct stat st;
        unsigned char *bytes;
        size_t offset;
        CFDataRef data;
        CFURLRef url;
        Boolean stale = false;
        errno = 0;
        entry = readdir(dir);
        if (!entry) { if (errno) fail(); break; }
        if (!valid_name(entry->d_name)) continue;
        if (++count > MAX_GRANTS) fail();
        fd = openat(dirfd(dir), entry->d_name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0 || fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
            st.st_size <= 0 || st.st_size > MAX_BOOKMARK ||
            (size_t)st.st_size > MAX_TOTAL - total) fail();
        total += (size_t)st.st_size;
        bytes = malloc((size_t)st.st_size);
        if (!bytes) fail();
        offset = 0;
        while (offset < (size_t)st.st_size) {
            ssize_t n = read(fd, bytes + offset, (size_t)st.st_size - offset);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) fail();
            offset += (size_t)n;
        }
        { unsigned char extra; ssize_t n;
          do { n = read(fd, &extra, 1); } while (n < 0 && errno == EINTR);
          if (n != 0 || close(fd) != 0) fail(); }
        data = CFDataCreate(kCFAllocatorDefault, bytes, (CFIndex)offset);
        free(bytes);
        if (!data) fail();
        url = CFURLCreateByResolvingBookmarkData(kCFAllocatorDefault, data,
            kCFURLBookmarkResolutionWithSecurityScope | kCFURLBookmarkResolutionWithoutUIMask |
            kCFURLBookmarkResolutionWithoutMountingMask, NULL, NULL, &stale, NULL);
        CFRelease(data);
        if (!url) continue;
        if (!stale && CFURLStartAccessingSecurityScopedResource(url))
            scopes[scope_count++] = url;
        else CFRelease(url);
    }
    if (closedir(dir) != 0) fail();
    if (atexit(cleanup) != 0) fail();
}
