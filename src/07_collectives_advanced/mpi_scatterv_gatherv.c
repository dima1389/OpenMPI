/*
 * File:
 *   mpi_scatterv_gatherv.c
 *
 * Purpose:
 *   Demonstrate irregular data distribution and collection using MPI_Scatterv and MPI_Gatherv.
 *
 * Description:
 *   This program generalizes the scatter/gather pattern to the case where N is not required to be
 *   divisible by the number of processes. Rank 0 distributes a global input array of length N using
 *   per-rank counts and displacements:
 *     - counts[r] specifies how many elements rank r receives
 *     - displs[r] specifies the starting index (offset) of rank r's block in the global array
 *
 *   Each rank receives its local block, performs a local computation (here: doubles each element),
 *   and the results are collected back to rank 0 using MPI_Gatherv with the same counts/displs.
 *
 * Key concepts:
 *   - Collective communication with irregular partition sizes
 *   - counts[] and displs[] as distribution metadata
 *   - Scatterv (one-to-all variable counts)
 *   - Gatherv (all-to-one variable counts)
 *   - Contiguous block decomposition with remainder handling
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Rank 0 parses/sets N and allocates global input/output arrays
 *   4) Broadcast N so all ranks can derive local buffer sizes
 *   5) Rank 0 constructs counts[] and displs[] for an uneven contiguous partition
 *   6) Each rank allocates local buffers sized by counts[rank]
 *   7) Scatterv distributes variable-sized blocks into local buffers
 *   8) Each rank computes on its local data
 *   9) Gatherv collects variable-sized result blocks into a global output array
 *  10) Rank 0 prints input/output and the distribution metadata
 *  11) Free resources and finalize MPI
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Scatterv
 *   - MPI_Gatherv
 *   - MPI_Abort
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_scatterv_gatherv.c -o mpi_scatterv_gatherv
 *
 * Run:
 *   mpirun -n <p> ./mpi_scatterv_gatherv [N]
 *
 * Notes:
 *   - This example uses int arrays for simplicity and deterministic output.
 *   - Output is printed only on rank 0 to avoid interleaved stdout.
 */

#include <stdio.h>    /* Provides printf and fprintf for output */
#include <stdlib.h>   /* Provides malloc, free, and strtol */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * build_counts_displs
 *
 * Builds counts[] and displs[] for a contiguous block decomposition of N elements across size ranks.
 * The first (N % size) ranks receive one extra element to distribute the remainder.
 *
 * Parameters:
 *   N      - total number of elements
 *   size   - number of ranks
 *   counts - output array of length size containing per-rank counts
 *   displs - output array of length size containing per-rank displacements
 */
static void build_counts_displs(int N, int size, int *counts, int *displs)
{
    int base = N / size; /* Computes the minimum number of elements per rank. */
    int rem  = N % size; /* Computes how many ranks receive one additional element. */

    /* Initialize the first displacement at the start of the global array. */
    displs[0] = 0; /* Sets the starting offset for rank 0. */

    /* Compute counts and displacements for each rank deterministically. */
    for (int r = 0; r < size; ++r) {
        counts[r] = base + ((r < rem) ? 1 : 0); /* Assigns base or base+1 elements depending on remainder. */

        /* Compute displacement for the next rank as a prefix sum of counts. */
        if (r > 0) {
            displs[r] = displs[r - 1] + counts[r - 1]; /* Sets the start index of rank r block. */
        }
    }
}

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
    int *A = NULL;            /* Stores the global input array (rank 0 only) */
    int *B = NULL;            /* Stores the global output array (rank 0 only) */
    int *local_A = NULL;      /* Stores the local input block on each rank */
    int *local_B = NULL;      /* Stores the local output block on each rank */
    int local_n = 0;          /* Stores the number of elements received by this rank */
    int *counts = NULL;       /* Stores per-rank receive/send counts (rank 0 only) */
    int *displs = NULL;       /* Stores per-rank displacements (rank 0 only) */

    /* Initialize the MPI execution environment to enable collective operations. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Configure N on rank 0 to define the global problem size. */
    if (rank == 0) {
        N = 17; /* Defines a default length that is intentionally not a multiple of common process counts. */

        /* Parse an optional command-line N to allow scaling experiments. */
        if (argc >= 2) {
            char *end = NULL;                 /* Holds parsing end pointer for validation. */
            long tmp = strtol(argv[1], &end, 10); /* Converts argv[1] to a long integer. */

            /* Validate that parsing consumed the full string and produced a positive N. */
            if (end == argv[1] || *end != '\0' || tmp <= 0) {
                fprintf(stderr, "Usage: %s [N]  (N must be a positive integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Terminates all ranks consistently on invalid input. */
            }

            N = (int)tmp; /* Assigns validated N as the global array length. */
        }
    }

    /* Broadcast N so all ranks can determine their expected local sizes. */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Allocate and build counts/displs on rank 0, since only the root needs them for Scatterv/Gatherv. */
    if (rank == 0) {
        counts = (int *)malloc((size_t)size * sizeof(int)); /* Allocates per-rank count array. */
        displs = (int *)malloc((size_t)size * sizeof(int)); /* Allocates per-rank displacement array. */

        /* Validate allocation to ensure root can drive the collective operations. */
        if (counts == NULL || displs == NULL) {
            fprintf(stderr, "Rank 0: ERROR: failed to allocate counts/displs\n"); /* Reports allocation failure on root. */
            MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because collective progress requires a valid root. */
        }

        /* Compute deterministic irregular partition metadata for N elements across size ranks. */
        build_counts_displs(N, size, counts, displs);
    }

    /* Determine the local receive count for this rank by broadcasting the counts array indirectly. */
    {
        int tmp_count = 0; /* Provides a temporary holder for the local count query. */

        /* Scatter the per-rank counts so each rank learns its local_n without requiring an Allgather. */
        MPI_Scatter(counts, 1, MPI_INT, &tmp_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        local_n = tmp_count; /* Stores this rank's local block length. */
    }

    /* Allocate local buffers sized to this rank's variable block length. */
    local_A = (int *)malloc((size_t)local_n * sizeof(int)); /* Allocates storage for the received input block. */
    local_B = (int *)malloc((size_t)local_n * sizeof(int)); /* Allocates storage for the computed output block. */

    /* Validate local allocations to prevent undefined behavior in communication or computation. */
    if ((local_n > 0) && (local_A == NULL || local_B == NULL)) {
        fprintf(stderr, "Rank %d: ERROR: failed to allocate local buffers (local_n=%d)\n", rank, local_n); /* Reports allocation failure with context. */
        MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because collectives require all ranks to participate. */
    }

    /* Allocate and initialize global arrays on rank 0, which acts as the Scatterv/Gatherv root. */
    if (rank == 0) {
        A = (int *)malloc((size_t)N * sizeof(int)); /* Allocates the global input array on the root. */
        B = (int *)malloc((size_t)N * sizeof(int)); /* Allocates the global output array on the root. */

        /* Validate global allocations to ensure root can participate correctly in collectives. */
        if (A == NULL || B == NULL) {
            fprintf(stderr, "Rank 0: ERROR: failed to allocate global arrays (N=%d)\n", N); /* Reports allocation failure on root. */
            MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because collective operations require a valid root buffer. */
        }

        /* Initialize global input with deterministic values to simplify verification. */
        for (int i = 0; i < N; ++i) {
            A[i] = i + 1; /* Assigns a simple increasing sequence for easy traceability. */
        }
    }

    /* Distribute variable-sized blocks of the global input array to all ranks. */
    MPI_Scatterv(A, counts, displs, MPI_INT, local_A, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    /* Perform local computation independently on each rank's block. */
    for (int i = 0; i < local_n; ++i) {
        local_B[i] = 2 * local_A[i]; /* Applies a simple linear transform to demonstrate local processing. */
    }

    /* Collect variable-sized result blocks from all ranks into the global output array on rank 0. */
    MPI_Gatherv(local_B, local_n, MPI_INT, B, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /* Print distribution metadata and arrays on rank 0 to verify correctness. */
    if (rank == 0) {
        printf("N=%d, size=%d\n", N, size); /* Reports the global problem size and number of ranks. */

        printf("counts: "); /* Labels the per-rank block sizes. */
        for (int r = 0; r < size; ++r) {
            printf("%d%s", counts[r], (r + 1 == size) ? "" : " "); /* Prints counts with spacing control. */
        }
        printf("\n"); /* Terminates the counts line. */

        printf("displs: "); /* Labels the per-rank starting offsets. */
        for (int r = 0; r < size; ++r) {
            printf("%d%s", displs[r], (r + 1 == size) ? "" : " "); /* Prints displacements with spacing control. */
        }
        printf("\n"); /* Terminates the displacements line. */

        printf("Input A: "); /* Labels the original input sequence. */
        for (int i = 0; i < N; ++i) {
            printf("%d%s", A[i], (i + 1 == N) ? "" : " "); /* Prints each element with spacing control. */
        }
        printf("\n"); /* Terminates the input line. */

        printf("Output B (2*A): "); /* Labels the computed output sequence. */
        for (int i = 0; i < N; ++i) {
            printf("%d%s", B[i], (i + 1 == N) ? "" : " "); /* Prints each element with spacing control. */
        }
        printf("\n"); /* Terminates the output line. */
    }

    /* Free local buffers on all ranks to avoid memory leaks. */
    free(local_A); /* Releases the local received input buffer. */
    free(local_B); /* Releases the local computed output buffer. */

    /* Free root-only buffers after collectives have completed. */
    if (rank == 0) {
        free(A);      /* Releases the global input array owned by the root. */
        free(B);      /* Releases the global output array owned by the root. */
        free(counts); /* Releases the counts array owned by the root. */
        free(displs); /* Releases the displacements array owned by the root. */
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
