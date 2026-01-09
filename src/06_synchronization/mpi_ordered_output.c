/*
 * File:
 *   mpi_ordered_output.c
 *
 * Purpose:
 *   Demonstrate deterministic, rank-ordered stdout printing using point-to-point token passing.
 *
 * Description:
 *   MPI processes execute concurrently, so naive printf calls from all ranks typically interleave
 *   nondeterministically. This program enforces a strict output order (rank 0, rank 1, ..., rank N-1)
 *   by passing a single-byte "token" along the rank chain using blocking MPI_Send/MPI_Recv.
 *
 *   Protocol:
 *     - Rank 0 prints immediately, then sends the token to rank 1.
 *     - Rank r (r > 0) blocks in MPI_Recv until it receives the token from rank r-1, then prints,
 *       then sends the token to rank r+1 (if it exists).
 *
 * Key concepts:
 *   - Deterministic output ordering
 *   - Synchronization using point-to-point communication
 *   - Token passing as a synchronization primitive
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Rank 0 prints and sends token to rank 1
 *   4) Each rank r>0 receives token, prints, and forwards token to rank r+1
 *   5) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Send
 *   - MPI_Recv
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_ordered_output.c -o mpi_ordered_output
 *
 * Run:
 *   mpirun -n <p> ./mpi_ordered_output
 *
 * Notes:
 *   - The token payload is intentionally minimal because only ordering matters.
 *   - This approach is simple and deterministic but serializes output, so it is not scalable for
 *     high-volume logging; it is intended for teaching and debugging.
 */

#include <stdio.h>   /* Provides printf and fflush for output */
#include <mpi.h>     /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                /* Stores the rank (ID) of the calling process */
    int size;                /* Stores the total number of processes */
    const int tag = 0;       /* Defines the message tag used for token matching */
    unsigned char token = 1; /* Defines a minimal token payload to enforce ordering */

    /* Initialize the MPI execution environment to enable point-to-point communication. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* For ranks other than 0, wait until the token arrives from the previous rank. */
    if (rank > 0) {
        int src = rank - 1; /* Defines the source rank from which the token must be received. */

        /* Block until the token is received to ensure all earlier ranks have printed. */
        MPI_Recv(&token, 1, MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* Print a single line from this rank after the ordering constraint is satisfied. */
    printf("Ordered output: rank %d of %d\n", rank, size); /* Emits deterministic rank-ordered output. */

    /* Flush stdout to reduce buffering-related reordering when stdout is redirected. */
    fflush(stdout); /* Forces the line to be written immediately to the output stream. */

    /* Forward the token to the next rank to allow it to print. */
    if (rank < size - 1) {
        int dst = rank + 1; /* Defines the destination rank that will print next. */

        /* Send the token to the next rank to grant permission to print. */
        MPI_Send(&token, 1, MPI_BYTE, dst, tag, MPI_COMM_WORLD);
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
