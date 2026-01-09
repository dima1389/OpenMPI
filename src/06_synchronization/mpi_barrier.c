/*
 * File:
 *   mpi_barrier.c
 *
 * Purpose:
 *   Demonstrate global synchronization using MPI_Barrier and its effect on timing measurements.
 *
 * Description:
 *   This program illustrates the semantics of MPI_Barrier by creating an intentional imbalance:
 *     - Rank 0 performs an artificial delay (a busy-wait loop) before entering the barrier.
 *     - All ranks call MPI_Barrier, which forces faster ranks to wait until the slowest rank arrives.
 *
 *   The program measures:
 *     - time spent before the barrier (including the artificial delay on rank 0)
 *     - time spent waiting inside the barrier (primarily on non-zero ranks)
 *     - time spent after the barrier in a second short workload
 *
 * Key concepts:
 *   - Barrier synchronization (collective)
 *   - Rank skew / load imbalance and its impact on waiting time
 *   - Correct placement of barriers for comparable timing measurements
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Synchronize ranks to align the start of the experiment
 *   4) Execute an imbalanced pre-barrier phase (rank 0 delays)
 *   5) Call MPI_Barrier and measure barrier wait duration
 *   6) Execute a balanced post-barrier phase on all ranks
 *   7) Gather and report maximum barrier wait time on rank 0
 *   8) Finalize MPI runtime
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
 *   mpicc mpi_barrier.c -o mpi_barrier
 *
 * Run:
 *   mpirun -n <p> ./mpi_barrier [delay_iterations]
 *
 * Notes:
 *   - The artificial delay uses a busy-wait loop to avoid OS sleep granularity issues.
 *   - Output is printed only by rank 0 to avoid interleaved stdout.
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
    int rank;                       /* Stores the rank (ID) of the calling process */
    int size;                       /* Stores the total number of processes */
    long long delay_iters = 0;      /* Stores artificial delay loop iterations for rank 0 */
    double t0 = 0.0;                /* Stores a generic timestamp used for interval timing */
    double t1 = 0.0;                /* Stores a generic timestamp used for interval timing */
    double pre_elapsed = 0.0;       /* Stores elapsed time measured for the pre-barrier phase */
    double barrier_elapsed = 0.0;   /* Stores time spent waiting inside MPI_Barrier */
    double post_elapsed = 0.0;      /* Stores elapsed time measured for the post-barrier phase */
    double max_barrier = 0.0;       /* Stores maximum barrier wait time across ranks (valid on rank 0) */
    double max_pre = 0.0;           /* Stores maximum pre-barrier phase time across ranks (valid on rank 0) */
    double max_post = 0.0;          /* Stores maximum post-barrier phase time across ranks (valid on rank 0) */

    /* Initialize the MPI execution environment to enable collective synchronization. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Select the artificial delay length on rank 0 to keep configuration centralized. */
    if (rank == 0) {
        delay_iters = 200000000LL; /* Defines a default delay that typically produces visible waiting on other ranks. */

        /* Parse an optional command-line delay iteration count to control the imbalance magnitude. */
        if (argc >= 2) {
            char *end = NULL; /* Holds parsing end pointer for validation. */
            long long tmp = strtoll(argv[1], &end, 10); /* Converts argv[1] to a signed 64-bit value. */

            /* Validate that parsing consumed the full string and produced a non-negative delay. */
            if (end == argv[1] || *end != '\0' || tmp < 0) {
                fprintf(stderr, "Usage: %s [delay_iterations]  (delay_iterations must be a non-negative integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Terminates all ranks consistently on invalid input. */
            }

            /* Commit the validated delay length as the broadcast value. */
            delay_iters = tmp; /* Sets the delay loop length from argv after validation. */
        }
    }

    /* Broadcast the delay parameter so all ranks have consistent experiment configuration. */
    MPI_Bcast(&delay_iters, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* Synchronize ranks so the pre-barrier timing starts from a comparable program point. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Start timing the pre-barrier phase, which will be imbalanced by design. */
    t0 = MPI_Wtime();

    /* Introduce an artificial delay on rank 0 to force other ranks to wait at the barrier. */
    if (rank == 0) {
        volatile double sink = 0.0; /* Creates a volatile target to inhibit dead-code elimination of the loop. */

        /* Execute a busy-wait style loop to consume CPU cycles deterministically. */
        for (long long i = 1; i <= delay_iters; ++i) {
            sink += 1.0 / (double)i; /* Performs arithmetic to prevent trivial loop optimization. */
        }

        (void)sink; /* Explicitly references sink to document intentional non-use beyond side effects. */
    }

    /* End timing of the pre-barrier phase for this rank. */
    t1 = MPI_Wtime();
    pre_elapsed = t1 - t0; /* Computes the pre-barrier elapsed time observed by this rank. */

    /* Measure time spent inside the barrier by timing immediately around MPI_Barrier. */
    t0 = MPI_Wtime();

    /* Block until all ranks have reached this synchronization point. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Stop timing after the barrier has released this rank. */
    t1 = MPI_Wtime();
    barrier_elapsed = t1 - t0; /* Computes the time this rank spent waiting inside the barrier. */

    /* Start timing of the post-barrier phase, which is balanced across ranks. */
    t0 = MPI_Wtime();

    /* Execute a short, balanced workload on all ranks to demonstrate synchronized progression. */
    {
        volatile double sink = 0.0; /* Creates a volatile target to inhibit dead-code elimination. */

        /* Execute a fixed-size loop to create measurable work after synchronization. */
        for (long long i = 1; i <= 20000000LL; ++i) {
            sink += 1.0 / (double)(i + rank + 1); /* Mixes rank into the denominator to avoid identical instruction streams. */
        }

        (void)sink; /* Explicitly references sink to document intentional non-use beyond side effects. */
    }

    /* Stop timing of the post-barrier phase. */
    t1 = MPI_Wtime();
    post_elapsed = t1 - t0; /* Computes the post-barrier elapsed time observed by this rank. */

    /* Reduce timing metrics to rank 0 using MPI_MAX to capture the slowest observed times. */
    MPI_Reduce(&pre_elapsed, &max_pre, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&barrier_elapsed, &max_barrier, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&post_elapsed, &max_post, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Print a summary on rank 0 to avoid nondeterministic interleaving of output. */
    if (rank == 0) {
        printf("MPI_Barrier demo on %d ranks\n", size); /* Reports the communicator size used for the experiment. */
        printf("Configured artificial delay iterations on rank 0: %lld\n", delay_iters); /* Reports the imbalance magnitude. */
        printf("Max pre-barrier phase time:    %.6f s\n", max_pre); /* Reports the slowest pre-barrier time (dominated by rank 0 delay). */
        printf("Max barrier wait time:         %.6f s\n", max_barrier); /* Reports the worst-case waiting time caused by synchronization. */
        printf("Max post-barrier phase time:   %.6f s\n", max_post); /* Reports the slowest post-barrier workload time. */
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
