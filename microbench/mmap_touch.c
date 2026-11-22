/* Time how long it takes to bring a file into memory, by memory-mapping it and touching every
 * page (the way llama.cpp loads model weights), or by reading it with read().
 *
 * The first pass over a file that is not in the page cache waits on the disk. A second pass finds
 * the pages already in memory. The ratio is the cost of a cold start.
 *
 * usage: mmap_touch FILE [--mode touch|read] [--stride 16384] [--advise none|willneed|sequential]
 * --advise passes a hint to madvise() right after mapping: willneed asks the kernel to start reading
 * the whole range at once, sequential says pages will be read in order.
 * touch mode reads one byte every `stride` bytes (default: one per page) and sums them, so the
 * checksum can be verified independently.
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: mmap_touch FILE [--mode touch|read] [--stride N]\n");
        return 2;
    }
    const char *path = argv[1];
    const char *mode = arg_value(argc, argv, "--mode", "touch");
    size_t stride = (size_t)atol(arg_value(argc, argv, "--stride", "16384"));
    const char *advise = arg_value(argc, argv, "--advise", "none");
    if (stride < 1) return 2;

    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) {
        perror("open");
        return 2;
    }
    size_t size = (size_t)st.st_size;
    uint64_t checksum = 0;
    uint64_t t0 = now_ns();
    if (!strcmp(mode, "touch")) {
        const unsigned char *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            perror("mmap");
            return 2;
        }
        if (!strcmp(advise, "willneed")) madvise((void *)p, size, MADV_WILLNEED);
        else if (!strcmp(advise, "sequential")) madvise((void *)p, size, MADV_SEQUENTIAL);
        for (size_t off = 0; off < size; off += stride) checksum += p[off];
        /* the timing covers the page faults, which happen inside the loop above */
        uint64_t t1 = now_ns();
        munmap((void *)p, size);
        printf("{\"program\":\"mmap_touch\",\"mode\":\"touch\",\"advise\":\"%s\",\"bytes\":%zu,\"stride\":%zu,\"seconds\":%.6f,\"gbs\":%.3f,\"checksum\":%llu}\n",
               advise, size, stride, (double)(t1 - t0) / 1e9, (double)size / ((double)(t1 - t0) / 1e9) / 1e9, (unsigned long long)checksum);
    } else {
        size_t chunk = 4u << 20;
        char *buf = aligned_alloc(16384, chunk);
        size_t total = 0;
        ssize_t got;
        while ((got = read(fd, buf, chunk)) > 0) total += (size_t)got;
        uint64_t t1 = now_ns();
        printf("{\"program\":\"mmap_touch\",\"mode\":\"read\",\"bytes\":%zu,\"seconds\":%.6f,\"gbs\":%.3f,\"checksum\":%llu}\n", total, (double)(t1 - t0) / 1e9,
               (double)total / ((double)(t1 - t0) / 1e9) / 1e9, (unsigned long long)total);
        free(buf);
    }
    close(fd);
    return 0;
}
