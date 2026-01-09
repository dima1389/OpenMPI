#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

/*
 * STRICT INTERPRETATION:
 * - The sent value is a two-digit natural number XY
 * - X (tens digit) is the sender's rank (must be a single digit: 0..9)
 * - Y (ones digit) is a random digit 0..9
 * - Each sender uses a different random digit per destination rank
 *
 * Therefore, this program REQUIRES: number of processes (size) <= 10.
 *
 * Communication pattern:
 * - Each rank prepares one integer per destination in sendbuf[dest]
 * - MPI_Alltoall exchanges 1 integer between every pair of ranks
 * - Each rank prints all values it received (excluding itself)
 */

static int random_digit(unsigned int *seed)
{
#if defined(_WIN32)
    (void)seed;
    return rand() % 10;
#else
    return (int)(rand_r(seed) % 10);
#endif
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Enforce "rank is the first digit" => rank must be 0..9 => size <= 10 */
    if (size > 10) {
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: This task requires size <= 10 so that each rank fits into one decimal digit.\n"
                    "You started %d processes.\n",
                    size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        /* Not reached */
    }

#if defined(_WIN32)
    /* rand() is global state; seed per rank */
    srand((unsigned)time(NULL) ^ (unsigned)(rank * 2654435761u));
#endif

    int *sendbuf = (int *)malloc((size_t)size * sizeof(int));
    int *recvbuf = (int *)malloc((size_t)size * sizeof(int));
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    /* Different RNG stream per rank (and per run) */
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(rank * 1103515245u);

    /* Prepare one message per destination */
    for (int dest = 0; dest < size; ++dest) {
        if (dest == rank) {
            /* Self-message not used; keep placeholder */
            sendbuf[dest] = -1;
        } else {
            int tens = rank;                 /* rank is guaranteed 0..9 */
            int ones = random_digit(&seed);  /* random 0..9 per destination */
            sendbuf[dest] = tens * 10 + ones;
        }
    }

    /* Exchange: recvbuf[src] is what we got from process 'src' */
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    /* Print in rank order to avoid interleaving */
    for (int r = 0; r < size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r == rank) {
            printf("Process %d received:", rank);
            for (int src = 0; src < size; ++src) {
                if (src == rank) continue;
                printf(" %d", recvbuf[src]);
            }
            printf("\n");
            fflush(stdout);
        }
    }

    free(sendbuf);
    free(recvbuf);

    MPI_Finalize();
    return 0;
}
