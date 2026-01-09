/*
 * File:
 *   mpi_send_recv.c
 *
 * Purpose:
 *   Demonstrate basic point-to-point communication using MPI_Send and MPI_Recv.
 *
 * Description:
 *   This program transfers a single integer message from rank 0 (sender)
 *   to rank 1 (receiver) using blocking point-to-point operations.
 *
 *   If the program is started with fewer than 2 processes, it prints an
 *   error message on rank 0 and aborts, because the communication pattern
 *   requires at least ranks 0 and 1 to exist.
 *
 * Key concepts:
 *   - Point-to-point communication
 *   - Blocking send/receive semantics
 *   - Message matching via (source, destination, tag, communicator)
 *   - MPI_Status for receive metadata
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Validate that at least 2 ranks exist
 *   4) Rank 0 sends an integer to rank 1
 *   5) Rank 1 receives the integer from rank 0
 *   6) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Send
 *   - MPI_Recv
 *   - MPI_Abort
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_send_recv.c -o mpi_send_recv
 *
 * Run:
 *   mpirun -n 2 ./mpi_send_recv
 */

#include <stdio.h>   /* Provides printf and fprintf for output */
#include <mpi.h>     /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;            /* Stores the rank (ID) of the calling process */
    int size;            /* Stores the total number of processes */
    const int tag = 0;   /* Defines a message tag used for matching send/receive */

    /* Initialize the MPI execution environment to enable MPI calls. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Enforce the precondition that the program needs at least two processes
     * so that rank 0 can send and rank 1 can receive. */
    if (size < 2) {
        /* Restrict the error message to rank 0 to avoid redundant output. */
        if (rank == 0) {
            fprintf(stderr, "ERROR: mpi_send_recv requires at least 2 processes (got %d)\n", size);
        }
        /* Abort all ranks in the communicator to terminate consistently. */
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Execute the sender role on rank 0 to demonstrate MPI_Send usage. */
    if (rank == 0) {
        int value = 12345;   /* Defines the integer payload to be transmitted */
        int dest = 1;        /* Selects the destination rank for the message */

        /* Send the integer payload to rank 1 using a blocking point-to-point call.
         * The tuple (dest, tag, MPI_COMM_WORLD) defines the message envelope. */
        MPI_Send(&value, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);

        /* Report that the message was sent, including payload and routing metadata. */
        printf("Rank %d sent value %d to rank %d (tag=%d)\n", rank, value, dest, tag);
    }

    /* Execute the receiver role on rank 1 to demonstrate MPI_Recv usage. */
    if (rank == 1) {
        int value = 0;           /* Provides storage for the received integer payload */
        int source = 0;          /* Selects the expected source rank for the message */
        MPI_Status status;       /* Receives metadata about the completed receive */

        /* Receive one integer from rank 0 using a blocking point-to-point call.
         * The tuple (source, tag, MPI_COMM_WORLD) must match the sender envelope. */
        MPI_Recv(&value, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

        /* Report the received payload and confirm the sender as seen in MPI_Status. */
        printf("Rank %d received value %d from rank %d (tag=%d)\n",
               rank, value, status.MPI_SOURCE, status.MPI_TAG);
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
