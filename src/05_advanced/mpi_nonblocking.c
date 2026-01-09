/*
 * File:
 *   mpi_nonblocking.c
 *
 * Purpose:
 *   Demonstrate non-blocking point-to-point communication with MPI_Isend and MPI_Irecv.
 *
 * Description:
 *   This program implements a ring exchange where each rank:
 *     - posts a non-blocking receive from its left neighbor
 *     - posts a non-blocking send to its right neighbor
 *     - performs a small local computation while communication progresses
 *     - completes communication using MPI_Waitall
 *
 *   The ring topology is defined as:
 *     left  = (rank - 1 + size) % size
 *     right = (rank + 1) % size
 *
 *   Each rank sends its own rank value as an integer payload and receives the
 *   left neighbor's rank value, which enables an unambiguous correctness check.
 *
 * Key concepts:
 *   - Non-blocking communication (progress + overlap)
 *   - Request handles (MPI_Request) as operation descriptors
 *   - Completion semantics (MPI_Wait / MPI_Waitall)
 *   - Avoiding deadlock via posted receives and non-blocking sends
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Compute left/right neighbors in a ring
 *   4) Post MPI_Irecv and MPI_Isend
 *   5) Perform local work while messages are in flight
 *   6) Complete communication with MPI_Waitall
 *   7) Validate and print results
 *   8) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Irecv
 *   - MPI_Isend
 *   - MPI_Waitall
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_nonblocking.c -o mpi_nonblocking
 *
 * Run:
 *   mpirun -n <p> ./mpi_nonblocking
 *
 * Notes:
 *   - Output order is not deterministic because ranks print concurrently.
 *   - The local "work" is a simple arithmetic loop intended only to illustrate overlap.
 */

#include <stdio.h>   /* Provides printf for output */
#include <mpi.h>     /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                  /* Stores the rank (ID) of the calling process */
    int size;                  /* Stores the total number of processes */
    int left;                  /* Stores the rank of the left neighbor in the ring */
    int right;                 /* Stores the rank of the right neighbor in the ring */
    const int tag = 0;         /* Defines a message tag used for matching send/receive */
    int send_value = 0;        /* Stores the integer payload sent to the right neighbor */
    int recv_value = -1;       /* Stores the integer payload received from the left neighbor */
    MPI_Request reqs[2];       /* Stores request handles for non-blocking operations */
    MPI_Status stats[2];       /* Stores completion metadata for the requests */

    /* Initialize the MPI execution environment to enable MPI calls. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute ring neighbors using modular arithmetic on the rank space. */
    left = (rank - 1 + size) % size; /* Selects the source rank for the incoming message. */
    right = (rank + 1) % size;       /* Selects the destination rank for the outgoing message. */

    /* Define the payload so correctness can be verified by simple rank comparison. */
    send_value = rank; /* Encodes sender identity directly in the message payload. */

    /* Post the non-blocking receive first so the receive is ready before matching sends arrive. */
    MPI_Irecv(&recv_value, 1, MPI_INT, left, tag, MPI_COMM_WORLD, &reqs[0]);

    /* Post the non-blocking send so the outgoing message can be injected without blocking. */
    MPI_Isend(&send_value, 1, MPI_INT, right, tag, MPI_COMM_WORLD, &reqs[1]);

    /* Perform local computation to illustrate potential communication/computation overlap. */
    {
        volatile double sink = 0.0; /* Creates a volatile accumulation target to inhibit dead-code elimination. */

        /* Execute a bounded loop with floating-point operations to occupy the CPU briefly. */
        for (long long i = 1; i <= 5000000LL; ++i) {
            sink += 1.0 / (double)i; /* Performs a division and accumulation as representative compute work. */
        }

        (void)sink; /* Explicitly references sink to document intentional non-use beyond side effects. */
    }

    /* Complete both non-blocking operations and ensure buffers are safe to reuse. */
    MPI_Waitall(2, reqs, stats);

    /* Validate that the received payload matches the expected left neighbor rank. */
    {
        int expected = left; /* Computes the payload expected from the left neighbor. */
        int ok = (recv_value == expected); /* Compares received payload against the expected value. */

        /* Print the result, including neighbor identities and correctness status. */
        printf("Rank %d: received %d from left %d, sent %d to right %d, status=%s\n",
               rank, recv_value, left, send_value, right, ok ? "OK" : "MISMATCH");
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
