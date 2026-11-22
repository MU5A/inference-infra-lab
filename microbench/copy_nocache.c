/* Copy a file without leaving its pages in the page cache, so the copy starts cold.
 *
 * On macOS this sets F_NOCACHE on both files. Elsewhere it flushes the copy and asks the kernel to
 * drop it. Whether the result is really cold is not assumed: the Day 2 run times the first and the
 * second pass over the copy, and rejects the experiment if they are not clearly different.
 *
 * usage: copy_nocache SRC DST
 */
#include <fcntl.h>
#include <unistd.h>

#include "common.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: copy_nocache SRC DST\n");
        return 2;
    }
    int in = open(argv[1], O_RDONLY), out = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (in < 0 || out < 0) {
        perror("open");
        return 2;
    }
#ifdef F_NOCACHE
    fcntl(in, F_NOCACHE, 1);
    fcntl(out, F_NOCACHE, 1);
#endif
    size_t chunk = 4u << 20;
    char *buf = aligned_alloc(16384, chunk);
    size_t total = 0;
    ssize_t got;
    while ((got = read(in, buf, chunk)) > 0) {
        ssize_t off = 0;
        while (off < got) {
            ssize_t w = write(out, buf + off, (size_t)(got - off));
            if (w < 0) {
                perror("write");
                return 2;
            }
            off += w;
        }
        total += (size_t)got;
    }
    fsync(out);
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
    posix_fadvise(in, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(in);
    close(out);
    printf("{\"program\":\"copy_nocache\",\"bytes\":%zu}\n", total);
    free(buf);
    return 0;
}
