/*
 * File:
 *   mpi_rank_size.c
 *
 * Purpose:
 *   Demonstrate how to query MPI process rank and communicator size.
 *
 * Description:
 *   This program initializes the MPI environment and retrieves:
 *     - the rank (unique identifier) of the calling process
 *     - the total number of processes in the communicator
 *
 *   The values are printed by every process to illustrate that:
 *     - rank differs per process
 *     - communicator size is identical across all processes
 *
 * Key concepts:
 *   - MPI rank
 *   - Communicator size
 *   - MPI_COMM_WORLD
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query process rank
 *   3) Query communicator size
 *   4) Output rank and size information
 *   5) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_rank_size.c -o mpi_rank_size
 *
 * Run:
 *   mpirun -n <num_processes> ./mpi_rank_size
 */

#include <stdio.h>   /* Provides printf for formatted output */
#include <mpi.h>     /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by each MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;    /* Stores the rank (ID) of the calling process */
    int size;    /* Stores the total number of processes */

    /* Initialize the MPI execution environment.
     * Required before any MPI communication or inquiry calls. */
    MPI_Init(&argc, &argv);

    /* Obtain the rank of this process in the global communicator.
     * Each process receives a unique integer in [0, size-1]. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Obtain the total number of processes in the global communicator.
     * This value is the same on all processes. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Print the rank and communicator size.
     * Output order is not guaranteed due to concurrent execution. */
    printf("Process rank: %d, Communicator size: %d\n", rank, size);

    /* Cleanly shut down the MPI execution environment.
     * No MPI calls are valid after this point. */
    MPI_Finalize();

    /* Indicate successful program termination. */
    return 0;
}
