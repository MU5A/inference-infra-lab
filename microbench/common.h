/* Shared helpers for the Day 2 microbenchmarks. */
#ifndef COMMON_H
#define COMMON_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Hint the scheduler about which kind of core to use. On Apple Silicon "interactive" prefers the
 * performance cores and "background" prefers the efficiency cores. macOS has no way to pin a thread
 * to a core, so this is a request, not a guarantee. "default" leaves the thread alone. Returns 0 on
 * success. On other systems only "default" succeeds. */
static inline int set_qos(const char *name) {
    if (!strcmp(name, "default")) return 0;
#ifdef __APPLE__
    if (!strcmp(name, "interactive")) return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (!strcmp(name, "background")) return pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
#endif
    return -1;
}

/* xorshift64*: small, fast and deterministic from a seed. */
static inline uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ull;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Median of n values. Sorts the array in place. */
static inline double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static inline const char *arg_value(int argc, char **argv, const char *flag, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], flag)) return argv[i + 1];
    return fallback;
}

static inline int arg_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], flag)) return 1;
    return 0;
}

#endif
