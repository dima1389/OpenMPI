/*
 * File:
 *   mpi_bcast_reduce.c
 *
 * Purpose:
 *   Demonstrate collective communication using MPI_Bcast and MPI_Reduce.
 *
 * Description:
 *   This program broadcasts a scalar value N from rank 0 to all ranks and then
 *   computes the global sum of a distributed arithmetic range using MPI_Reduce.
 *
 *   Work distribution:
 *     - The integers 1..N are partitioned into contiguous blocks across ranks.
 *     - Each rank computes a local partial sum over its assigned subrange.
 *     - MPI_Reduce with MPI_SUM combines partial sums into the global sum on rank 0.
 *
 * Key concepts:
 *   - Collective communication
 *   - Broadcast (one-to-all)
 *   - Reduction (many-to-one)
 *   - Deterministic decomposition of a global index space
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Rank 0 sets N (from argv or default)
 *   4) Broadcast N to all ranks
 *   5) Each rank computes its local subrange and local sum
 *   6) Reduce all local sums into a global sum on rank 0
 *   7) Rank 0 prints the result and (optionally) verifies analytically
 *   8) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Reduce
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_bcast_reduce.c -o mpi_bcast_reduce
 *
 * Run:
 *   mpirun -n <p> ./mpi_bcast_reduce [N]
 *
 * Notes:
 *   - The result fits in 64-bit signed integer for reasonably sized N.
 *   - Output order is controlled by printing only on rank 0.
 */

#include <stdio.h>    /* Provides printf and fprintf for output */
#include <stdlib.h>   /* Provides strtoll for command-line parsing */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                /* Stores the rank (ID) of the calling process */
    int size;                /* Stores the total number of processes */
    long long N = 0;         /* Stores the global upper bound of the summation */
    long long local_sum = 0; /* Stores this rank's partial sum */
    long long global_sum = 0;/* Stores the reduced global sum on rank 0 */

    /* Initialize the MPI execution environment to enable collective operations. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Select N on rank 0 to define the global problem size. */
    if (rank == 0) {
        /* Use a conservative default so the example runs without arguments. */
        N = 100; /* Defines the default summation bound when argv does not provide one. */

        /* Parse an optional command-line N to make the program parameterized. */
        if (argc >= 2) {
            char *end = NULL;                     /* Holds parsing end pointer for validation. */
            long long tmp = strtoll(argv[1], &end, 10); /* Converts argv[1] to a signed 64-bit value. */

            /* Validate that parsing consumed the full string and produced a non-negative bound. */
            if (end == argv[1] || *end != '\0' || tmp < 0) {
                fprintf(stderr, "Usage: %s [N]  (N must be a non-negative integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1);     /* Terminates all ranks consistently on invalid input. */
            }

            /* Commit the validated command-line value as the broadcast value. */
            N = tmp; /* Sets the global bound from argv after validation. */
        }
    }

    /* Broadcast N from rank 0 to all ranks so every process has the same problem size. */
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* Compute a deterministic contiguous block partition of the range [1..N]. */
    {
        long long base = N / size;        /* Computes the minimum number of items per rank. */
        long long rem  = N % size;        /* Computes the number of ranks that receive one extra item. */

        /* Assign rank-local block length using a block-cyclic remainder distribution. */
        long long local_n = base + ((rank < rem) ? 1 : 0); /* Computes this rank's block size. */

        /* Compute the starting index for this rank's block using prefix sums of block sizes. */
        long long start = 1 + rank * base + ((rank < rem) ? rank : rem); /* Computes first integer in the local block. */

        /* Compute the ending index for this rank's block, preserving emptiness when local_n is zero. */
        long long end = (local_n > 0) ? (start + local_n - 1) : 0; /* Computes last integer in the local block. */

        /* Accumulate the local partial sum over the assigned inclusive interval [start..end]. */
        for (long long i = start; i <= end; ++i) {
            local_sum += i; /* Adds each local term to the partial sum to prepare for reduction. */
        }
    }

    /* Reduce all local partial sums into a single global sum on rank 0 using integer addition. */
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Print the final result on rank 0 to avoid interleaved output. */
    if (rank == 0) {
        long long expected = (N * (N + 1)) / 2; /* Computes the closed-form reference sum for verification. */

        /* Report the reduced sum and the reference value to demonstrate correctness. */
        printf("Sum(1..%lld) via MPI_Reduce = %lld (expected %lld)\n", N, global_sum, expected);
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
