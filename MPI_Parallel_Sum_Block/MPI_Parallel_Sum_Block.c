#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * MPI sum of first N natural numbers using block (contiguous range) decomposition.
 *
 * Goal:
 *   Compute S = 1 + 2 + ... + N
 *
 * Block decomposition:
 *   Distribute the integer interval [1, N] into 'size' contiguous blocks.
 *   Each rank computes its local sum over [local_start, local_end], then:
 *     MPI_Reduce(local_sum, global_sum, MPI_SUM, root=0)
 *
 * Notes:
 *  - Works for any N >= 0 and any number of processes.
 *  - Handles the remainder when N is not divisible by 'size' by distributing
 *    one extra element to the first 'remainder' ranks.
 */

int main(int argc, char *argv[])
{
    int rank, size;
    long long N = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Input: either command line or interactive (rank 0 only). */
    if (rank == 0) {
        if (argc >= 2) {
            char *end = NULL;
            long long tmp = strtoll(argv[1], &end, 10);
            if (end == argv[1] || *end != '\0' || tmp < 0) {
                fprintf(stderr, "Usage: %s <N>  (N must be a non-negative integer)\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            N = tmp;
        } else {
            printf("Enter N (non-negative integer): ");
            fflush(stdout);
            if (scanf("%lld", &N) != 1 || N < 0) {
                fprintf(stderr, "Invalid input. N must be a non-negative integer.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    /* Broadcast N so every rank knows the problem size. */
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /*
     * Compute each rank's block [local_start, local_end] within [1, N].
     *
     * Let:
     *   q = N / size           (base block size)
     *   r = N % size           (remainder)
     *
     * Ranks 0..(r-1) get (q+1) elements, remaining ranks get q elements.
     */
    long long q = (size > 0) ? (N / size) : 0;
    long long r = (size > 0) ? (N % size) : 0;

    long long local_count = (rank < r) ? (q + 1) : q;

    /* Number of elements assigned to ranks smaller than me (prefix sum). */
    long long prefix = rank * q + (rank < r ? rank : r);

    long long local_start = 1 + prefix;                 /* inclusive */
    long long local_end   = local_start + local_count - 1; /* inclusive */

    /* Local sum using arithmetic series formula on the local interval. */
    long long local_sum = 0;
    if (local_count > 0) {
        /* sum_{k=a..b} k = (a + b) * (b - a + 1) / 2 */
        long long a = local_start;
        long long b = local_end;
        long long cnt = local_count;
        local_sum = (a + b) * cnt / 2;
    }

    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum(1..%lld) = %lld\n", N, global_sum);
    }

    MPI_Finalize();
    return 0;
}
