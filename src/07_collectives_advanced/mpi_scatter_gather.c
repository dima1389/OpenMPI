/*
 * File:
 *   mpi_scatter_gather.c
 *
 * Purpose:
 *   Demonstrate block-wise data distribution and collection using MPI_Scatter and MPI_Gather.
 *
 * Description:
 *   This program demonstrates a common SPMD data-parallel pattern:
 *     - Rank 0 initializes an input array of length N.
 *     - The array is distributed in equal contiguous blocks to all ranks via MPI_Scatter.
 *     - Each rank performs a local computation on its block (here: squares each element).
 *     - The transformed blocks are collected back to rank 0 via MPI_Gather.
 *
 *   Constraint:
 *     - N must be divisible by the number of processes, because MPI_Scatter/MPI_Gather
 *       require equal send counts per destination in this simple form.
 *
 * Key concepts:
 *   - Collective communication
 *   - Equal-sized block decomposition
 *   - Scatter (one-to-all distribution)
 *   - Gather (all-to-one collection)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Rank 0 parses/sets N and allocates global input/output arrays
 *   4) Broadcast N so all ranks can size local buffers
 *   5) Validate N % size == 0
 *   6) Allocate per-rank local buffers
 *   7) Scatter global input array into local blocks
 *   8) Compute locally on each block
 *   9) Gather local results back into a global output array
 *  10) Rank 0 prints input and output for verification
 *  11) Free resources and finalize MPI
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Scatter
 *   - MPI_Gather
 *   - MPI_Abort
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_scatter_gather.c -o mpi_scatter_gather
 *
 * Run:
 *   mpirun -n <p> ./mpi_scatter_gather [N]
 *
 * Notes:
 *   - Output printing is limited to rank 0 to avoid interleaved stdout.
 *   - This example uses int arrays for simplicity and deterministic output.
 */

#include <stdio.h>    /* Provides printf and fprintf for output */
#include <stdlib.h>   /* Provides malloc, free, and strtol */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                 /* Stores the rank (ID) of the calling process */
    int size;                 /* Stores the total number of processes */
    int N = 0;                /* Stores the global array length */
    int local_n = 0;          /* Stores the number of elements handled by each rank */
    int *A = NULL;            /* Stores the global input array (rank 0 only) */
    int *B = NULL;            /* Stores the global output array (rank 0 only) */
    int *local_A = NULL;      /* Stores the local input block on each rank */
    int *local_B = NULL;      /* Stores the local output block on each rank */

    /* Initialize the MPI execution environment to enable collective operations. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Configure N on rank 0 to define the global problem size. */
    if (rank == 0) {
        N = 16; /* Defines a default array length suitable for small demonstrations. */

        /* Parse an optional command-line N to allow scaling experiments. */
        if (argc >= 2) {
            char *end = NULL;                 /* Holds parsing end pointer for validation. */
            long tmp = strtol(argv[1], &end, 10); /* Converts argv[1] to a long integer. */

            /* Validate that parsing consumed the full string and produced a positive N. */
            if (end == argv[1] || *end != '\0' || tmp <= 0) {
                fprintf(stderr, "Usage: %s [N]  (N must be a positive integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Terminates all ranks consistently on invalid input. */
            }

            /* Store the validated N in int range for this example. */
            N = (int)tmp; /* Assigns the validated size to the global length. */
        }
    }

    /* Broadcast N so all ranks can allocate local buffers consistently. */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Enforce the equal-block precondition required by MPI_Scatter/MPI_Gather in this form. */
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: N (%d) must be divisible by number of processes (%d)\n", N, size); /* Reports the constraint violation. */
        }
        MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks to avoid undefined behavior in collectives. */
    }

    /* Compute the per-rank block size based on equal contiguous partitioning. */
    local_n = N / size; /* Computes the number of elements each rank will receive. */

    /* Allocate per-rank input and output buffers sized to the local block. */
    local_A = (int *)malloc((size_t)local_n * sizeof(int)); /* Allocates storage for the scattered input block. */
    local_B = (int *)malloc((size_t)local_n * sizeof(int)); /* Allocates storage for the locally computed output block. */

    /* Validate local allocations to prevent null dereferences in communication or computation. */
    if (local_A == NULL || local_B == NULL) {
        fprintf(stderr, "Rank %d: ERROR: failed to allocate local buffers\n", rank); /* Reports allocation failure with rank context. */
        MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because progress requires all ranks to participate. */
    }

    /* Allocate and initialize global arrays on rank 0, which acts as the scatter/gather root. */
    if (rank == 0) {
        A = (int *)malloc((size_t)N * sizeof(int)); /* Allocates the global input array on the root. */
        B = (int *)malloc((size_t)N * sizeof(int)); /* Allocates the global output array on the root. */

        /* Validate global allocations to ensure root can participate correctly in collectives. */
        if (A == NULL || B == NULL) {
            fprintf(stderr, "Rank 0: ERROR: failed to allocate global arrays\n"); /* Reports allocation failure on root. */
            MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because collective operations require a valid root buffer. */
        }

        /* Initialize the global input array with deterministic values for verification. */
        for (int i = 0; i < N; ++i) {
            A[i] = i + 1; /* Assigns a simple increasing sequence to simplify correctness checks. */
        }
    }

    /* Distribute equal-sized blocks of the global input array to all ranks. */
    MPI_Scatter(A, local_n, MPI_INT, local_A, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    /* Perform local computation independently on each rank's block. */
    for (int i = 0; i < local_n; ++i) {
        local_B[i] = local_A[i] * local_A[i]; /* Squares each element to demonstrate a local transform. */
    }

    /* Collect equal-sized result blocks from all ranks into the global output array on rank 0. */
    MPI_Gather(local_B, local_n, MPI_INT, B, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    /* Print the input and output arrays on rank 0 to confirm correct distribution and collection. */
    if (rank == 0) {
        printf("Input A: "); /* Labels the original input sequence. */
        for (int i = 0; i < N; ++i) {
            printf("%d%s", A[i], (i + 1 == N) ? "" : " "); /* Prints each element with spacing control. */
        }
        printf("\n"); /* Terminates the input line. */

        printf("Output B (A[i]^2): "); /* Labels the computed output sequence. */
        for (int i = 0; i < N; ++i) {
            printf("%d%s", B[i], (i + 1 == N) ? "" : " "); /* Prints each element with spacing control. */
        }
        printf("\n"); /* Terminates the output line. */
    }

    /* Free local buffers on all ranks to avoid memory leaks. */
    free(local_A); /* Releases the local scattered input buffer. */
    free(local_B); /* Releases the local computed output buffer. */

    /* Free global buffers on rank 0 after collectives have completed. */
    if (rank == 0) {
        free(A); /* Releases the global input array owned by the root. */
        free(B); /* Releases the global output array owned by the root. */
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
