#pragma once

// Shim for missing timespec_get in older sysroots/libc headers.
// Force-included via CMake to satisfy <ctime> using-declaration.
#include <time.h>

#ifndef TIME_UTC
#define TIME_UTC 1
#endif

#if !defined(timespec_get)
extern "C" inline int timespec_get(struct timespec* ts, int base) {
    if (!ts || base != TIME_UTC) {
        return 0;
    }
#if defined(CLOCK_REALTIME)
    if (clock_gettime(CLOCK_REALTIME, ts) == 0) {
        return base;
    }
#endif
    return 0;
}
#endif