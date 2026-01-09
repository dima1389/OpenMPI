/*
 * File:
 *   mpi_timing_max.c
 *
 * Purpose:
 *   Demonstrate timing measurement on each MPI rank and reduction of the maximum elapsed time.
 *
 * Description:
 *   This program measures a simple, deterministic workload on each MPI process using MPI_Wtime().
 *   The per-rank elapsed times are then reduced with MPI_Reduce(MPI_MAX) so that rank 0 receives
 *   the maximum elapsed time across all ranks.
 *
 *   In SPMD programs, the maximum rank time is typically the relevant performance metric because
 *   total wall-clock completion time is bounded by the slowest rank.
 *
 * Key concepts:
 *   - Wall-clock timing with MPI_Wtime
 *   - Process synchronization with MPI_Barrier
 *   - Reduction with MPI_MAX to obtain the slowest rank time
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Synchronize ranks before timing (MPI_Barrier)
 *   4) Measure a local workload using MPI_Wtime()
 *   5) Reduce local elapsed times using MPI_MAX into rank 0
 *   6) Rank 0 prints the maximum elapsed time
 *   7) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Barrier
 *   - MPI_Wtime
 *   - MPI_Reduce
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_timing_max.c -o mpi_timing_max
 *
 * Run:
 *   mpirun -n <p> ./mpi_timing_max [iterations]
 *
 * Notes:
 *   - The workload is a floating-point accumulation loop that is long enough to be measurable.
 *   - A volatile sink is used to prevent aggressive dead-code elimination of the loop.
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
    int rank;                    /* Stores the rank (ID) of the calling process */
    int size;                    /* Stores the total number of processes */
    long long iterations = 0;    /* Stores the number of loop iterations for the workload */
    double t_start = 0.0;        /* Stores local start timestamp in seconds */
    double t_end = 0.0;          /* Stores local end timestamp in seconds */
    double local_elapsed = 0.0;  /* Stores elapsed time measured on this rank */
    double max_elapsed = 0.0;    /* Stores maximum elapsed time across ranks (valid on rank 0) */

    /* Initialize the MPI execution environment to enable timing and collectives. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Select the workload size on rank 0 to keep program configuration centralized. */
    if (rank == 0) {
        iterations = 200000000LL; /* Defines a default iteration count that typically yields measurable timing. */

        /* Parse an optional command-line iteration count to allow experiment control. */
        if (argc >= 2) {
            char *end = NULL; /* Holds parsing end pointer for validation. */
            long long tmp = strtoll(argv[1], &end, 10); /* Converts argv[1] to a signed 64-bit value. */

            /* Validate that parsing consumed the full string and produced a positive iteration count. */
            if (end == argv[1] || *end != '\0' || tmp <= 0) {
                fprintf(stderr, "Usage: %s [iterations]  (iterations must be a positive integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Terminates all ranks consistently on invalid input. */
            }

            /* Commit the validated iteration count as the broadcast value. */
            iterations = tmp; /* Sets the workload size from argv after validation. */
        }
    }

    /* Broadcast the iteration count so all ranks execute the same nominal workload. */
    MPI_Bcast(&iterations, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* Synchronize all ranks so timing begins from a comparable program point. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Record the start time immediately before entering the measured workload. */
    t_start = MPI_Wtime();

    /* Execute a simple floating-point workload to produce measurable elapsed time. */
    {
        volatile double sink = 0.0; /* Creates a volatile accumulation target to inhibit dead-code elimination. */

        /* Accumulate a deterministic series so each iteration performs non-trivial arithmetic. */
        for (long long i = 1; i <= iterations; ++i) {
            sink += 1.0 / (double)i; /* Performs a division and accumulation to model compute work. */
        }

        (void)sink; /* Explicitly references sink to document intentional non-use beyond side effects. */
    }

    /* Record the end time immediately after finishing the measured workload. */
    t_end = MPI_Wtime();

    /* Compute the local elapsed time for this rank in seconds. */
    local_elapsed = t_end - t_start;

    /* Reduce all ranks' elapsed times into rank 0 using MPI_MAX to obtain the slowest rank time. */
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Print summary information on rank 0 to avoid multi-rank output interleaving. */
    if (rank == 0) {
        printf("MPI timing (max across %d ranks): iterations=%lld, max_elapsed=%.6f s\n",
               size, iterations, max_elapsed); /* Reports the globally relevant elapsed time metric. */
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
