/*
 * File:
 *   mpi_vector_mul.c
 *
 * Purpose:
 *   Demonstrate block-distributed element-wise vector multiplication using MPI collectives.
 *
 * Description:
 *   This example demonstrates data-parallel SPMD execution where rank 0 generates two vectors,
 *   evenly distributes contiguous blocks to all ranks with MPI_Scatter, each rank computes
 *   C[i] = A[i] * B[i] for its local block, and rank 0 collects the full result with MPI_Gather.
 *   Observable outcome: correct element-wise product vector printed by rank 0.
 *
 * Key concepts:
 *   - ranks, MPI_COMM_WORLD, collective communication
 *   - blocking collectives, deterministic work partitioning
 *   - performance effect: balanced workload (requires N divisible by P)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and determine rank/size
 *   2) On rank 0, allocate and initialize full vectors A and B
 *   3) Scatter contiguous blocks of A and B to all ranks
 *   4) Compute local element-wise products into local_C
 *   5) Gather local_C blocks to rank 0 as full vector C and print results
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Scatter, MPI_Gather
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_vector_mul.c -o mpi_vector_mul
 *
 * Execution:
 *   mpiexec -n <P> ./mpi_vector_mul
 *
 * Inputs:
 *   - Command-line arguments: none (N is a fixed constant in the source)
 *   - Interactive input: none
 *
 * References:
 *   - MPI Standard (Collectives): MPI_Scatter, MPI_Gather
 */

#include <stdio.h>   // Provides printf() for formatted output.
#include <stdlib.h>  // Provides malloc(), free(), srand(), rand(), and general utilities.
#include <mpi.h>     // Declares the MPI API used for parallel execution and communication.
#include <time.h>    // Provides time() used to seed the pseudo-random generator.

int main(int argc, char *argv[])                         // Defines program entry point with MPI-compatible argc/argv.
{
    int rank, size;                                      // Declares this process rank and the total communicator size.
    int N = 16;                                          // Defines global vector length; must be evenly divisible by size for block partitioning.
    int local_n;                                         // Declares per-rank block length (number of elements handled locally).

    double *A = NULL;                                    // Declares root-owned pointer for full input vector A (allocated only on rank 0).
    double *B = NULL;                                    // Declares root-owned pointer for full input vector B (allocated only on rank 0).
    double *C = NULL;                                    // Declares root-owned pointer for full output vector C (allocated only on rank 0).

    double *local_A;                                     // Declares per-rank pointer for the local block of A received via scatter.
    double *local_B;                                     // Declares per-rank pointer for the local block of B received via scatter.
    double *local_C;                                     // Declares per-rank pointer for the local block of C computed locally.

    MPI_Init(&argc, &argv);                              // Initializes the MPI runtime so MPI calls are valid on all ranks.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                // Queries this process rank within MPI_COMM_WORLD for SPMD control flow.
    MPI_Comm_size(MPI_COMM_WORLD, &size);                // Queries the number of ranks to size the block distribution.

    if (N % size != 0) {                                 // Validates that N can be evenly partitioned into equal contiguous blocks.
        if (rank == 0)                                   // Restricts the diagnostic message to rank 0 to avoid duplicated output.
            printf("Vector size N must be divisible by number of processes.\n"); // Reports the partitioning constraint violation.
        MPI_Finalize();                                  // Shuts down MPI cleanly since the program cannot proceed correctly.
        return 1;                                        // Returns a non-zero status to indicate failure to the invoking environment.
    }

    local_n = N / size;                                  // Computes the fixed block size each rank will receive and process.

    local_A = (double *)malloc(local_n * sizeof(double)); // Allocates storage for this rank's contiguous block of A elements.
    local_B = (double *)malloc(local_n * sizeof(double)); // Allocates storage for this rank's contiguous block of B elements.
    local_C = (double *)malloc(local_n * sizeof(double)); // Allocates storage for this rank's contiguous block of output products.

    if (rank == 0) {                                     // Selects rank 0 as the root responsible for global allocation and initialization.
        A = (double *)malloc(N * sizeof(double));         // Allocates the full input vector A on the root for subsequent scattering.
        B = (double *)malloc(N * sizeof(double));         // Allocates the full input vector B on the root for subsequent scattering.
        C = (double *)malloc(N * sizeof(double));         // Allocates the full output vector C on the root for gathering results.

        srand((unsigned int)time(NULL));                  // Seeds the C PRNG once on the root so generated inputs vary between runs.
        for (int i = 0; i < N; i++) {                     // Iterates over all global indices to initialize full input vectors.
            A[i] = rand() % 10;                           // Assigns A[i] a small integer value (0..9) representable as double.
            B[i] = rand() % 10;                           // Assigns B[i] a small integer value (0..9) representable as double.
        }
    }

    MPI_Scatter(A, local_n, MPI_DOUBLE,                   // Sends contiguous blocks from root's A to all ranks (including root).
                local_A, local_n, MPI_DOUBLE,             // Receives this rank's block into local_A with matching count and datatype.
                0, MPI_COMM_WORLD);                       // Specifies root rank 0 and communicator MPI_COMM_WORLD for the collective.

    MPI_Scatter(B, local_n, MPI_DOUBLE,                   // Sends contiguous blocks from root's B to all ranks (including root).
                local_B, local_n, MPI_DOUBLE,             // Receives this rank's block into local_B with matching count and datatype.
                0, MPI_COMM_WORLD);                       // Specifies root rank 0 and communicator MPI_COMM_WORLD for the collective.

    for (int i = 0; i < local_n; i++) {                   // Iterates over the local block indices assigned to this rank.
        local_C[i] = local_A[i] * local_B[i];             // Computes element-wise product for this rank's slice of the vectors.
    }

    MPI_Gather(local_C, local_n, MPI_DOUBLE,              // Contributes this rank's local_C block to the root in rank order.
               C, local_n, MPI_DOUBLE,                    // Receives all blocks into C on the root with the same fixed block size.
               0, MPI_COMM_WORLD);                        // Specifies root rank 0 and communicator MPI_COMM_WORLD for the collective.

    if (rank == 0) {                                      // Restricts final output and root-only deallocation to rank 0.
        printf("Vector A:\n");                            // Prints a label for the first input vector.
        for (int i = 0; i < N; i++)                       // Iterates over the full vector length to print all A elements.
            printf("%5.1f ", A[i]);                       // Prints A[i] with fixed width and one decimal for aligned readability.
        printf("\n\n");                                   // Terminates the line and inserts an extra blank line for separation.

        printf("Vector B:\n");                            // Prints a label for the second input vector.
        for (int i = 0; i < N; i++)                       // Iterates over the full vector length to print all B elements.
            printf("%5.1f ", B[i]);                       // Prints B[i] with fixed width and one decimal for aligned readability.
        printf("\n\n");                                   // Terminates the line and inserts an extra blank line for separation.

        printf("Vector C = A * B:\n");                    // Prints a label describing the computed result.
        for (int i = 0; i < N; i++)                       // Iterates over the full vector length to print all C elements.
            printf("%5.1f ", C[i]);                       // Prints C[i] with fixed width and one decimal for aligned readability.
        printf("\n");                                     // Terminates the final output line.

        free(A);                                          // Releases the root-owned allocation for A to avoid a memory leak.
        free(B);                                          // Releases the root-owned allocation for B to avoid a memory leak.
        free(C);                                          // Releases the root-owned allocation for C to avoid a memory leak.
    }

    free(local_A);                                        // Releases this rank's local A block buffer.
    free(local_B);                                        // Releases this rank's local B block buffer.
    free(local_C);                                        // Releases this rank's local C block buffer.

    MPI_Finalize();                                       // Finalizes the MPI runtime and releases MPI-managed resources.
    return 0;                                             // Returns success status to the operating system.
}
