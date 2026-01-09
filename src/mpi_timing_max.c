/*
 * File:
 *   mpi_timing_max.c
 *
 * Purpose:
 *   Demonstrate MPI wall-clock timing and runtime characterization via a max-reduction.
 *
 * Description:
 *   This example measures per-rank elapsed time with MPI_Wtime() after synchronizing all
 *   ranks with MPI_Barrier(). It then computes the global parallel runtime as the maximum
 *   local elapsed time using MPI_Reduce(..., MPI_MAX, root=0).
 *   Observable outcome: the reported maximum time matches the slowest rank under load imbalance.
 *
 * Key concepts:
 *   - ranks, communicators, collectives
 *   - blocking collective synchronization (MPI_Barrier) and blocking collective reduction (MPI_Reduce)
 *   - performance effect: load imbalance (slowest rank determines SPMD time-to-solution)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and discover rank/size
 *   2) Synchronize ranks before timing
 *   3) Perform rank-dependent work and measure local elapsed time
 *   4) Reduce local times with MPI_MAX to obtain effective parallel runtime (rank 0 prints)
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init
 *   - MPI_Finalize
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Barrier
 *   - MPI_Wtime
 *   - MPI_Reduce
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_timing_max.c -o mpi_timing_max
 *
 * Execution:
 *   mpiexec -n <P> mpi_timing_max
 *
 * Inputs:
 *   - Command-line arguments: none
 *   - Interactive input: none
 *
 * References:
 *   - MPI Standard: Collective synchronization (MPI_Barrier), reduction (MPI_Reduce), timing (MPI_Wtime)
 */

#include <stdio.h>   // Provides printf() for reporting timing results.                            /* required for output */
#include <mpi.h>     // Provides MPI_Init/MPI_Finalize and collective/timing MPI APIs.             /* required for MPI */

/* Program entry point with standard MPI signature to receive argc/argv from mpiexec. */
int main(int argc, char *argv[])
{
    int rank;                      // Stores this process's rank in MPI_COMM_WORLD.               /* identifies caller */
    int size;                      // Stores total number of ranks in MPI_COMM_WORLD.             /* for reporting */

    double local_start;            // Stores per-rank start timestamp from MPI_Wtime().            /* timing start */
    double local_finish;           // Stores per-rank finish timestamp from MPI_Wtime().           /* timing end */
    double local_elapsed;          // Stores per-rank elapsed time (finish - start).               /* local duration */
    double elapsed;                // Stores global maximum elapsed time on root after reduction.  /* SPMD runtime */

    MPI_Init(&argc, &argv);        // Initializes MPI runtime; must precede almost all MPI calls.  /* required init */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Queries caller rank within the global communicator.   /* rank id */
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Queries communicator size (number of processes).      /* world size */

    MPI_Barrier(MPI_COMM_WORLD);   // Synchronizes ranks so timing begins from a common point.     /* fair timing */

    local_start = MPI_Wtime();     // Captures the start time using MPI's wall-clock timer.        /* start timestamp */

    volatile double dummy = 0.0;   // Prevents optimizing away the workload by creating a side effect. /* keep loop */
    for (long i = 0; i < (rank + 1) * 10000000L; i++) // Runs rank-scaled iterations to induce imbalance. /* workload */
    {
        dummy += i * 0.0000001;    // Performs floating-point work to consume time measurably.     /* compute */
    }

    local_finish = MPI_Wtime();    // Captures the finish time after the workload completes.       /* end timestamp */

    local_elapsed = local_finish - local_start; // Computes elapsed wall time for this rank.       /* duration */

    printf("Process %d: local elapsed time = %f seconds\n", rank, local_elapsed); // Reports local time (may interleave). /* local print */

    MPI_Reduce(&local_elapsed,     // Supplies each rank's local elapsed time as the reduction input. /* sendbuf */
               &elapsed,           // Receives the reduced value on root (undefined on non-root).    /* recvbuf */
               1,                  // Reduces exactly one double value per rank.                     /* count */
               MPI_DOUBLE,         // Declares the element datatype as double.                        /* type */
               MPI_MAX,            // Selects maximum to model SPMD runtime dominated by slowest rank. /* op */
               0,                  // Sets rank 0 as the root that receives the reduced result.      /* root */
               MPI_COMM_WORLD);    // Performs the reduction across all ranks in the global communicator. /* comm */

    if (rank == 0)                 // Restricts printing of the global result to the root rank.     /* single output */
    {
        printf("\nMaximum elapsed time across %d processes: %f seconds\n", size, elapsed); // Reports effective parallel runtime. /* max print */
    }

    MPI_Finalize();                // Shuts down the MPI runtime and releases MPI-managed resources. /* required finalize */

    return 0;                      // Returns success status to the host environment.              /* normal exit */
}
