/*
 * File:
 *   mpi_sum_stride.c
 *
 * Purpose:
 *   Demonstrate cyclic (strided) work distribution across MPI ranks and global reduction.
 *
 * Description:
 *   This example computes the arithmetic series S = 1 + 2 + ... + n in parallel.
 *   Rank 0 reads n interactively, broadcasts it to all ranks, each rank computes a
 *   partial sum over indices i = rank+1, rank+1+P, ... (cyclic distribution), and
 *   the final result is obtained with MPI_Reduce (MPI_SUM) on rank 0.
 *   Runtime is measured with MPI_Wtime; the maximum per-rank duration is reported.
 *
 * Key concepts:
 *   - ranks, communicators, collectives
 *   - deterministic, blocking behavior
 *   - load balancing via cyclic (strided) partitioning
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and read/broadcast input n
 *   2) Synchronize ranks and start timing
 *   3) Compute local partial sum with cyclic index assignment
 *   4) Reduce partial sums and reduce maximum elapsed time; print on rank 0
 *
 * MPI features used:
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Bcast, MPI_Barrier, MPI_Reduce, MPI_Wtime
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_sum_stride.c -o mpi_sum_stride
 *
 * Execution:
 *   mpiexec -n <P> mpi_sum_stride
 *
 * Inputs:
 *   - Command-line arguments: none
 *   - Interactive input: n (rank 0 only; non-negative integer)
 *
 * References:
 *   - MPI Standard (MPI-4.x): Collective communication (Bcast/Reduce), Synchronization (Barrier), Timing (Wtime)
 */

#include <stdio.h>   // Declares printf, fprintf, scanf, fflush, stderr for console I/O and diagnostics.
#include <stdlib.h>  // Declares exit for process termination on unrecoverable input errors (rank 0).
#include <mpi.h>     // Declares the MPI API for initialization, collectives, and timing.

/* Reads a non-negative integer n from stdin; only rank 0 calls this function. */
static long long get_input_rank0(void)
{
    long long n = 0;                                     // Holds the parsed upper bound of the series on rank 0.
    int items = 0;                                       // Captures scanf()'s conversion count to validate input.

    printf("Enter n (non-negative integer): ");          // Prompts the user for the series upper bound.
    fflush(stdout);                                      // Flushes stdout to ensure the prompt is visible before blocking for input.
    items = scanf("%lld", &n);                           // Parses a signed 64-bit integer from stdin into n.

    if (items != 1 || n < 0) {                           // Validates that parsing succeeded and the value is within the expected domain.
        fprintf(stderr, "ERROR: n must be a non-negative integer.\n"); // Reports invalid input to stderr for immediate visibility.
        exit(1);                                         // Terminates rank 0 because continuing would propagate an invalid problem size.
    }

    return n;                                            // Returns the validated series upper bound to the caller (rank 0).
}

int main(int argc, char *argv[])
{
    int csize = 0;                                       // Stores the total number of MPI processes in MPI_COMM_WORLD.
    int prank = 0;                                       // Stores the rank ID of this process within MPI_COMM_WORLD.
    long long n = 0;                                     // Stores the global series upper bound (broadcast from rank 0).
    long long local_sum = 0;                             // Accumulates this rank's partial sum of the series.
    long long total_sum = 0;                             // Receives the final reduced sum on rank 0 (undefined on other ranks).

    double t_start = 0.0;                                // Stores the per-rank start timestamp in seconds (MPI_Wtime timebase).
    double t_end = 0.0;                                  // Stores the per-rank end timestamp in seconds (MPI_Wtime timebase).
    double t_local = 0.0;                                // Stores this rank's elapsed time for the measured region.
    double t_max = 0.0;                                  // Receives the maximum elapsed time across ranks on rank 0.

    MPI_Init(&argc, &argv);                              // Initializes the MPI runtime; must precede most other MPI calls.
    MPI_Comm_size(MPI_COMM_WORLD, &csize);               // Queries the communicator size to parameterize cyclic distribution.
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);               // Queries this process's rank to select its cyclic index subset.

    if (prank == 0) {                                    // Restricts interactive input to rank 0 to avoid duplicated prompts.
        n = get_input_rank0();                           // Reads and validates n on the designated root rank.
    }

    MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);  // Broadcasts n so every rank computes on a consistent problem size.

    MPI_Barrier(MPI_COMM_WORLD);                         // Synchronizes ranks so timing excludes skew from earlier ranks reaching the region sooner.
    t_start = MPI_Wtime();                               // Captures the start time after synchronization for a comparable timing window.

    for (long long i = (long long)prank + 1; i <= n; i += (long long)csize) { // Iterates over this rank's cyclic indices in [1..n].
        local_sum += i;                                  // Accumulates the partial sum contribution for each assigned term.
    }

    t_end = MPI_Wtime();                                 // Captures the end time immediately after local computation completes.
    t_local = t_end - t_start;                           // Computes this rank's elapsed time for the measured region.

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // Sums partial results onto rank 0.
    MPI_Reduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);         // Computes the worst-case (maximum) elapsed time on rank 0.

    if (prank == 0) {                                    // Ensures only the root rank prints the single definitive output.
        printf("S = 1 + 2 + ... + %lld = %lld\n", n, total_sum); // Reports the computed arithmetic series result.
        printf("Elapsed time (max across ranks): %.6f s\n", t_max); // Reports the SPMD-relevant runtime dominated by the slowest rank.
    }

    MPI_Finalize();                                      // Finalizes the MPI runtime and releases MPI resources before program exit.
    return 0;                                            // Returns a success status code to the operating system.
}
