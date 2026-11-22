/* Memory latency by pointer chasing.
 *
 * A working set of N nodes, one cache line each, is linked into a single random cycle. Following the
 * chain makes every load depend on the one before it, so the hardware cannot overlap or prefetch
 * them, and time per step is the true load-to-use latency of whichever level of the hierarchy holds
 * the working set. Sweeping the size from a few KiB to a GiB shows each level as a plateau.
 *
 * The random cycle is built with Sattolo's algorithm, which produces one cycle covering all nodes
 * (a plain shuffle can leave short cycles that would silently shrink the working set). --selftest
 * checks that, and checks the pointer chain itself.
 *
 * usage: chase [--min-kib 4] [--max-mib 1024] [--per-octave 2] [--node-bytes 128]
 *              [--qos interactive] [--seed 1] [--selftest]
 */
#include <math.h>

#include "common.h"

static volatile uintptr_t sink;

static void sattolo(uint32_t *next, size_t n, uint64_t seed) {
    uint64_t state = seed ? seed : 88172645463325252ull;
    for (size_t i = 0; i < n; i++) next[i] = (uint32_t)i;
    for (size_t i = n - 1; i >= 1; i--) {
        size_t j = (size_t)(rng_next(&state) % i); /* 0 .. i-1, never i itself */
        uint32_t t = next[i];
        next[i] = next[j];
        next[j] = t;
    }
}

static int is_single_cycle(const uint32_t *next, size_t n) {
    size_t steps = 0, i = 0;
    do {
        i = next[i];
        steps++;
    } while (i != 0 && steps <= n);
    return i == 0 && steps == n;
}

/* Follow the pointer chain and count distinct nodes visited before returning to the start. */
static size_t chain_length(char *base, size_t n) {
    char *start = base, *p = base;
    size_t steps = 0;
    do {
        p = *(char **)p;
        steps++;
    } while (p != start && steps <= n);
    return steps;
}

static char *build_chain(size_t n, size_t node_bytes, uint64_t seed) {
    uint32_t *next = malloc(n * sizeof(uint32_t));
    void *mem = NULL;
    /* posix_memalign, not aligned_alloc: macOS refuses sizes that are not a multiple of the alignment. */
    if (!next || posix_memalign(&mem, 16384, n * node_bytes) != 0) {
        free(next);
        return NULL;
    }
    char *base = mem;
    sattolo(next, n, seed);
    for (size_t i = 0; i < n; i++) *(char **)(base + i * node_bytes) = base + (size_t)next[i] * node_bytes;
    free(next);
    return base;
}

static double time_steps(char *start, uint64_t steps) {
    char *p = start;
    uint64_t t0 = now_ns();
    for (uint64_t i = 0; i < steps; i++) p = *(char **)p;
    uint64_t t1 = now_ns();
    sink = (uintptr_t)p;
    return (double)(t1 - t0);
}

static int selftest(void) {
    for (size_t n = 2; n <= 5000; n = n * 3 / 2 + 1) {
        uint32_t *next = malloc(n * sizeof(uint32_t));
        sattolo(next, n, 7 + n);
        int ok = is_single_cycle(next, n);
        free(next);
        if (!ok) {
            fprintf(stderr, "not a single cycle for n=%zu\n", n);
            return 1;
        }
        char *base = build_chain(n, 128, 11 + n);
        if (!base) {
            fprintf(stderr, "allocation failed in selftest for n=%zu\n", n);
            return 1;
        }
        size_t len = chain_length(base, n);
        free(base);
        if (len != n) {
            fprintf(stderr, "chain visits %zu of %zu nodes\n", len, n);
            return 1;
        }
    }
    printf("{\"program\":\"chase\",\"selftest\":\"ok\"}\n");
    return 0;
}

int main(int argc, char **argv) {
    if (arg_flag(argc, argv, "--selftest")) return selftest();
    double min_kib = atof(arg_value(argc, argv, "--min-kib", "4"));
    double max_mib = atof(arg_value(argc, argv, "--max-mib", "1024"));
    int per_octave = atoi(arg_value(argc, argv, "--per-octave", "2"));
    size_t node_bytes = (size_t)atol(arg_value(argc, argv, "--node-bytes", "128"));
    const char *qos = arg_value(argc, argv, "--qos", "interactive");
    uint64_t seed = (uint64_t)atoll(arg_value(argc, argv, "--seed", "1"));
    if (per_octave < 1 || node_bytes < 8 || min_kib <= 0 || max_mib <= 0) {
        fprintf(stderr, "bad arguments\n");
        return 2;
    }
    int qos_status = set_qos(qos);

    printf("{\"program\":\"chase\",\"node_bytes\":%zu,\"qos\":\"%s\",\"qos_applied\":%s,\"seed\":%llu,\"points\":[", node_bytes,
           qos, qos_status == 0 ? "true" : "false", (unsigned long long)seed);
    double lo = min_kib * 1024.0, hi = max_mib * 1024.0 * 1024.0;
    size_t previous = 0;
    int first = 1;
    for (int k = 0;; k++) {
        double wanted = lo * pow(2.0, (double)k / per_octave);
        if (wanted > hi * 1.0000001) break;
        size_t n = (size_t)(wanted / (double)node_bytes);
        if (n < 2) n = 2;
        size_t bytes = n * node_bytes;
        if (bytes == previous) continue;
        previous = bytes;

        char *base = build_chain(n, node_bytes, seed + (uint64_t)k);
        if (!base) {
            fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
            return 2;
        }
        /* Touch every node twice so caches, TLBs and page tables are populated before timing. */
        time_steps(base, n);
        time_steps(base, n);
        /* Choose a step count that makes one measurement last at least 30 ms. */
        uint64_t steps = 1u << 16;
        while (time_steps(base, steps) < 30e6 && steps < (1ull << 31)) steps *= 2;
        double ns[5];
        for (int r = 0; r < 5; r++) ns[r] = time_steps(base, steps) / (double)steps;
        double fastest = ns[0];
        for (int r = 1; r < 5; r++)
            if (ns[r] < fastest) fastest = ns[r];
        double med = median(ns, 5);
        printf("%s{\"bytes\":%zu,\"ns_median\":%.3f,\"ns_min\":%.3f,\"steps\":%llu}", first ? "" : ",", bytes, med, fastest,
               (unsigned long long)steps);
        fflush(stdout);
        first = 0;
        free(base);
    }
    printf("]}\n");
    return 0;
}
