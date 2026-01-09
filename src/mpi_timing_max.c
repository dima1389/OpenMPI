#include <stdio.h>
#include <mpi.h>

/*
 * This example demonstrates:
 *  - Measuring execution time locally on each MPI process
 *  - Synchronizing processes using MPI_Barrier
 *  - Computing the maximum execution time across all processes
 *
 * The maximum elapsed time is typically the relevant metric in
 * SPMD (Single Program Multiple Data) parallel programs, because
 * overall execution time is bounded by the slowest rank.
 */

int main(int argc, char *argv[])
{
    int rank;     // Rank of the current process
    int size;     // Total number of processes

    double local_start;    // Local start timestamp
    double local_finish;   // Local finish timestamp
    double local_elapsed;  // Elapsed time on this process
    double elapsed;        // Maximum elapsed time (computed on rank 0)

    /* Initialize the MPI execution environment */
    MPI_Init(&argc, &argv);

    /* Obtain communicator information */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Synchronize all processes before starting the timer.
     * This ensures that no process starts measuring time
     * before others have reached the same execution point.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Start local timer */
    local_start = MPI_Wtime();

    /*
     * ------------------------------------------------------
     * Simulated workload
     * ------------------------------------------------------
     * Each process executes a loop whose duration depends
     * on its rank. This intentionally creates load imbalance
     * so that different ranks take different amounts of time.
     */
    volatile double dummy = 0.0;
    for (long i = 0; i < (rank + 1) * 10000000L; i++)
    {
        dummy += i * 0.0000001;
    }
    /*
     * ------------------------------------------------------
     */

    /* Stop local timer */
    local_finish = MPI_Wtime();

    /* Compute local elapsed time */
    local_elapsed = local_finish - local_start;

    /*
     * Print local timing information for each process.
     * This output is not synchronized and may appear interleaved.
     */
    printf("Process %d: local elapsed time = %f seconds\n",
           rank, local_elapsed);

    /*
     * Reduce all local elapsed times to a single value on rank 0.
     * MPI_MAX is used to obtain the maximum execution time
     * across all processes.
     */
    MPI_Reduce(&local_elapsed,
               &elapsed,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    /*
     * The maximum elapsed time corresponds to the slowest process
     * and therefore represents the effective parallel runtime.
     */
    if (rank == 0)
    {
        printf("\nMaximum elapsed time across %d processes: %f seconds\n",
               size, elapsed);
    }

    /* Finalize the MPI execution environment */
    MPI_Finalize();

    return 0;
}
