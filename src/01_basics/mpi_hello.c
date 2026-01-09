/*
 * File:
 *   mpi_hello.c
 *
 * Purpose:
 *   Demonstrate basic MPI program structure and process identification.
 *
 * Description:
 *   This program initializes the MPI execution environment, determines
 *   the rank of each process and the total number of processes in the
 *   global communicator, and prints a greeting message from each rank.
 *
 * Key concepts:
 *   - MPI process
 *   - Rank
 *   - Communicator (MPI_COMM_WORLD)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Print a message identifying each process
 *   4) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_hello.c -o mpi_hello
 *
 * Run:
 *   mpirun -n <num_processes> ./mpi_hello
 */

#include <stdio.h>   /* Provides printf for formatted output */
#include <mpi.h>     /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point of the MPI program.
 * Every MPI process executes this function independently.
 */
int main(int argc, char *argv[])
{
    int rank;        /* Holds the unique rank (ID) of this process */
    int size;        /* Holds the total number of processes */

    /* Initialize the MPI execution environment.
     * This must be called before any other MPI function. */
    MPI_Init(&argc, &argv);

    /* Determine the rank of this process within MPI_COMM_WORLD.
     * Ranks are consecutive integers in the range [0, size-1]. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes in MPI_COMM_WORLD.
     * This value is identical on all processes. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Print a greeting message that uniquely identifies this process
     * by its rank and reports the total communicator size. */
    printf("Hello from MPI process %d out of %d processes\n", rank, size);

    /* Finalize the MPI execution environment.
     * After this call, no MPI function may be used. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
