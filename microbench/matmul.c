/* Single-precision matrix multiply, four ways, to show what loop order, blocking and a tuned
 * library each buy.
 *
 *   naive      i-j-k: the inner loop walks a column of B, striding a whole row between loads
 *   ikj        i-k-j: the inner loop walks rows of B and C, which are contiguous
 *   blockedNN  i-k-j inside NN x NN tiles, so a tile of B stays in cache while it is reused
 *   accelerate Apple's Accelerate framework (cblas_sgemm). Library-managed threads and the matrix
 *              hardware, so it is not a like-for-like single-core comparison. Set
 *              VECLIB_MAXIMUM_THREADS=1 to limit it to one thread.
 *
 * Every variant is checked against a double-precision reference. A variant that is fast because it
 * is wrong fails the check, and the program exits non-zero.
 *
 * usage: matmul [--sizes 128,256,512,1024] [--reps 3] [--blocks 32,64]
 *               [--variants naive,ikj,blocked,accelerate] [--naive-max 1024] [--qos interactive]
 *               [--selftest]
 */
#include <math.h>

#include "common.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

static void mm_naive(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
}

static void mm_ikj(int n, const float *A, const float *B, float *C) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

static void mm_blocked(int n, const float *A, const float *B, float *C, int bs) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int ii = 0; ii < n; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs) {
                int imax = ii + bs < n ? ii + bs : n, kmax = kk + bs < n ? kk + bs : n, jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        float a = A[(size_t)i * n + k];
                        for (int j = jj; j < jmax; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
                    }
            }
}

#ifdef __APPLE__
static void mm_accelerate(int n, const float *A, const float *B, float *C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);
}
#endif

static void reference(int n, const float *A, const float *B, double *R) {
    memset(R, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) R[(size_t)i * n + j] += a * (double)B[(size_t)k * n + j];
        }
}

static double rel_error(int n, const float *C, const double *R) {
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        double d = (double)C[i] - R[i];
        num += d * d;
        den += R[i] * R[i];
    }
    return sqrt(num / den);
}

typedef struct {
    char name[32];
    int kind; /* 0 naive, 1 ikj, 2 blocked, 3 accelerate */
    int bs;
} variant_t;

static void run_variant(const variant_t *v, int n, const float *A, const float *B, float *C) {
    switch (v->kind) {
    case 0: mm_naive(n, A, B, C); break;
    case 1: mm_ikj(n, A, B, C); break;
    case 2: mm_blocked(n, A, B, C, v->bs); break;
#ifdef __APPLE__
    case 3: mm_accelerate(n, A, B, C); break;
#endif
    default: break;
    }
}

static void fill(float *m, size_t count, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < count; i++) m[i] = (float)((rng_next(&s) >> 40) / 16777216.0) * 2.0f - 1.0f;
}

static int parse_list(const char *text, int *out, int max) {
    int count = 0;
    char *copy = strdup(text), *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok && count < max; tok = strtok_r(NULL, ",", &save)) out[count++] = atoi(tok);
    free(copy);
    return count;
}

#define TOLERANCE 1e-4

int main(int argc, char **argv) {
    int selftest = arg_flag(argc, argv, "--selftest");
    int sizes[16], blocks[8];
    int nsizes = parse_list(arg_value(argc, argv, "--sizes", selftest ? "48,100" : "128,256,512,1024"), sizes, 16);
    int nblocks = parse_list(arg_value(argc, argv, "--blocks", selftest ? "16,32" : "32,64"), blocks, 8);
    int reps = atoi(arg_value(argc, argv, "--reps", selftest ? "1" : "3"));
    int naive_max = atoi(arg_value(argc, argv, "--naive-max", "1024"));
    const char *want = arg_value(argc, argv, "--variants", "naive,ikj,blocked,accelerate");
    const char *qos = arg_value(argc, argv, "--qos", "interactive");
    if (reps < 1 || nsizes < 1) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }
    int qos_status = set_qos(qos);

    variant_t variants[16];
    int nv = 0;
    if (strstr(want, "naive")) variants[nv++] = (variant_t){"naive", 0, 0};
    if (strstr(want, "ikj")) variants[nv++] = (variant_t){"ikj", 1, 0};
    if (strstr(want, "blocked"))
        for (int b = 0; b < nblocks; b++) {
            variants[nv] = (variant_t){"", 2, blocks[b]};
            snprintf(variants[nv].name, sizeof variants[nv].name, "blocked%d", blocks[b]);
            nv++;
        }
#ifdef __APPLE__
    if (strstr(want, "accelerate")) variants[nv++] = (variant_t){"accelerate", 3, 0};
#endif

    int all_ok = 1;
    printf("{\"program\":\"matmul\",\"qos\":\"%s\",\"qos_applied\":%s,\"tolerance\":%g,\"results\":[", qos, qos_status == 0 ? "true" : "false", TOLERANCE);
    for (int s = 0; s < nsizes; s++) {
        int n = sizes[s];
        size_t count = (size_t)n * n;
        float *A = aligned_alloc(16384, ((count * 4 + 16383) / 16384) * 16384), *B = aligned_alloc(16384, ((count * 4 + 16383) / 16384) * 16384),
              *C = aligned_alloc(16384, ((count * 4 + 16383) / 16384) * 16384);
        double *R = malloc(count * sizeof(double));
        fill(A, count, 1000 + (uint64_t)n);
        fill(B, count, 2000 + (uint64_t)n);
        reference(n, A, B, R);

        printf("%s{\"n\":%d,\"variants\":{", s ? "," : "", n);
        int printed = 0;
        for (int v = 0; v < nv; v++) {
            if (variants[v].kind == 0 && n > naive_max) continue;
            run_variant(&variants[v], n, A, B, C); /* warm-up, and the first result checked */
            double err = rel_error(n, C, R);
            int ok = err < TOLERANCE;
            if (!ok) all_ok = 0;
            double rates[64];
            for (int r = 0; r < reps && r < 64; r++) {
                uint64_t t0 = now_ns();
                run_variant(&variants[v], n, A, B, C);
                rates[r] = 2.0 * (double)n * n * n / ((double)(now_ns() - t0) / 1e9) / 1e9;
            }
            int m = reps < 64 ? reps : 64;
            /* The warm-up result was checked above. Check the last timed result too, so a variant
             * that is right on its first call and wrong afterwards (state left in C) cannot pass. */
            double err_last = rel_error(n, C, R);
            if (err_last > err) err = err_last;
            ok = err < TOLERANCE;
            if (!ok) all_ok = 0;
            double best = 0;
            for (int r = 0; r < m; r++)
                if (rates[r] > best) best = rates[r];
            double med = median(rates, m);
            printf("%s\"%s\":{\"gflops_best\":%.3f,\"gflops_median\":%.3f,\"rel_err\":%.3e,\"ok\":%s}", printed ? "," : "", variants[v].name, best, med, err,
                   ok ? "true" : "false");
            printed = 1;
        }
        printf("}}");
        free(A);
        free(B);
        free(C);
        free(R);
    }
    printf("]}\n");
    return all_ok ? 0 : 1;
}
