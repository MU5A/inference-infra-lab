/* Sustained memory bandwidth, in the style of STREAM.
 *
 * Four kernels over three arrays that are far larger than any cache:
 *   copy  c = a          scale b = s*c          add  c = a + b          triad a = b + s*c
 * Bytes moved are counted the STREAM way (reads plus writes, ignoring write-allocate traffic), so
 * the figures are comparable with published STREAM results. The arrays are checked afterwards
 * against the values the same sequence produces on scalars, so a kernel that did not run, or ran
 * on the wrong data, fails loudly instead of reporting a fast number.
 *
 * usage: membw [--array-mib 256] [--threads 1] [--reps 10] [--qos interactive] [--selftest]
 */
#include <math.h>
#include <pthread.h>

#include "common.h"

enum { COPY, SCALE, ADD, TRIAD, NKERNELS };
static const char *KERNEL_NAME[NKERNELS] = {"copy", "scale", "add", "triad"};
static const int BYTES_PER_ELEMENT[NKERNELS] = {16, 16, 24, 24};
#define SCALAR 3.0

typedef struct {
    double *a, *b, *c;
    size_t lo, hi;
    int kernel;
    const char *qos;
} job_t;

static void *worker(void *arg) {
    job_t *j = arg;
    set_qos(j->qos);
    double *restrict a = j->a, *restrict b = j->b, *restrict c = j->c;
    switch (j->kernel) {
    case COPY:
        for (size_t i = j->lo; i < j->hi; i++) c[i] = a[i];
        break;
    case SCALE:
        for (size_t i = j->lo; i < j->hi; i++) b[i] = SCALAR * c[i];
        break;
    case ADD:
        for (size_t i = j->lo; i < j->hi; i++) c[i] = a[i] + b[i];
        break;
    case TRIAD:
        for (size_t i = j->lo; i < j->hi; i++) a[i] = b[i] + SCALAR * c[i];
        break;
    }
    return NULL;
}

/* Run one kernel over the whole array on `threads` threads. Returns elapsed seconds. */
static double run_kernel(int kernel, double *a, double *b, double *c, size_t n, int threads, const char *qos) {
    pthread_t tid[64];
    job_t jobs[64];
    uint64_t t0 = now_ns();
    for (int t = 0; t < threads; t++) {
        jobs[t] = (job_t){a, b, c, n * (size_t)t / (size_t)threads, n * (size_t)(t + 1) / (size_t)threads, kernel, qos};
        pthread_create(&tid[t], NULL, worker, &jobs[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    return (double)(now_ns() - t0) / 1e9;
}

static int validate(const double *a, const double *b, const double *c, size_t n, int reps) {
    double aj = 1.0, bj = 2.0, cj = 0.0;
    for (int k = 0; k < reps; k++) {
        cj = aj;
        bj = SCALAR * cj;
        cj = aj + bj;
        aj = bj + SCALAR * cj;
    }
    for (size_t i = 0; i < n; i++) {
        if (fabs(a[i] - aj) > 1e-12 * fabs(aj) || fabs(b[i] - bj) > 1e-12 * fabs(bj) ||
            fabs(c[i] - cj) > 1e-12 * fabs(cj))
            return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int selftest = arg_flag(argc, argv, "--selftest");
    size_t mib = (size_t)atol(arg_value(argc, argv, "--array-mib", selftest ? "1" : "256"));
    int threads = atoi(arg_value(argc, argv, "--threads", selftest ? "2" : "1"));
    int reps = atoi(arg_value(argc, argv, "--reps", selftest ? "3" : "10"));
    const char *qos = arg_value(argc, argv, "--qos", "interactive");
    if (threads < 1 || threads > 64 || reps < 2 || mib < 1) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }

    size_t n = mib * 1024 * 1024 / sizeof(double);
    size_t bytes = n * sizeof(double);
    double *a = aligned_alloc(16384, bytes), *b = aligned_alloc(16384, bytes), *c = aligned_alloc(16384, bytes);
    if (!a || !b || !c) {
        fprintf(stderr, "allocation of 3 x %zu MiB failed\n", mib);
        return 2;
    }
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    double seconds[NKERNELS][256];
    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < NKERNELS; k++) seconds[k][r] = run_kernel(k, a, b, c, n, threads, qos);
    }
    int ok = validate(a, b, c, n, reps);

    printf("{\"program\":\"membw\",\"threads\":%d,\"qos\":\"%s\",\"array_mib\":%zu,\"reps\":%d,", threads, qos, mib, reps);
    printf("\"validated\":%s,\"kernels\":{", ok ? "true" : "false");
    for (int k = 0; k < NKERNELS; k++) {
        /* The first repetition warms the pages and is not counted. */
        double rates[256];
        int m = 0;
        for (int r = 1; r < reps; r++) rates[m++] = (double)BYTES_PER_ELEMENT[k] * (double)n / seconds[k][r] / 1e9;
        double best = 0, worst = 1e300;
        for (int i = 0; i < m; i++) {
            if (rates[i] > best) best = rates[i];
            if (rates[i] < worst) worst = rates[i];
        }
        double med = median(rates, m);
        printf("%s\"%s\":{\"gbs_median\":%.3f,\"gbs_best\":%.3f,\"gbs_min\":%.3f}", k ? "," : "", KERNEL_NAME[k], med, best, worst);
    }
    printf("}}\n");
    free(a);
    free(b);
    free(c);
    return ok ? 0 : 1;
}
