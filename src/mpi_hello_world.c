#include <stdio.h>      // Standard I/O library for printf
#include <string.h>     // For strlen used when sending the string
#include <mpi.h>        // MPI library header

// Maximum length (in chars) of the greeting string buffer, including the '\0'.
const int MAX_STRING = 100;

int main(void)
{
    char gret[MAX_STRING];  // Buffer for the greeting message (for send/recv)
    int csize;              // Total number of processes in MPI_COMM_WORLD
    int prank;              // Rank (ID) of this process in MPI_COMM_WORLD

    // Initialize the MPI execution environment.
    // Both arguments are NULL since we do not need to propagate argc/argv.
    MPI_Init(NULL, NULL);

    // Determine the number of processes in the communicator MPI_COMM_WORLD.
    // Result is stored in csize.
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    // Determine the rank (ID) of the calling process in MPI_COMM_WORLD.
    // Rank values are in the range [0, csize-1].
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // All processes with rank != 0 act as senders.
    if (prank != 0) {
        // Format a process-specific greeting message into the buffer.
        // The buffer must be large enough to hold the resulting string plus '\0'.
        sprintf(gret, "Greets from process %d of %d!", prank, csize);

        // Send the greeting string to process 0.
        // - gret: start address of the send buffer
        // - strlen(gret)+1: number of characters to send, including the '\0'
        // - MPI_CHAR: datatype of each element in the buffer
        // - 0: destination rank (root process)
        // - 0: message tag (arbitrary but must match in MPI_Recv)
        // - MPI_COMM_WORLD: communicator over which to send
        MPI_Send(gret, strlen(gret) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    } else {
        // Rank 0: acts as the root/collector process.

        // Print this process's own greeting locally (no communication involved).
        printf("Greets from process %d of %d!\n", prank, csize);

        // Loop over all other ranks [1, csize-1] and receive one message from each.
        for (int q = 1; q < csize; q++) {

            // Receive a null-terminated C string from process q.
            // - gret: receive buffer
            // - MAX_STRING: maximum number of MPI_CHAR elements to receive
            // - MPI_CHAR: datatype of each element
            // - q: source rank (the sender)
            // - 0: message tag (must match the tag in MPI_Send)
            // - MPI_COMM_WORLD: communicator
            // - MPI_STATUS_IGNORE: we do not inspect the status (source, tag, count)
            MPI_Recv(gret, MAX_STRING, MPI_CHAR, q, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Print the received message. The buffer is expected to contain a valid
            // null-terminated C string because the sender transmitted strlen+1 bytes.
            printf("%s\n", gret);
        }
    }

    // Cleanly shut down the MPI environment.
    // No MPI calls are allowed after MPI_Finalize.
    MPI_Finalize();

    // By convention, return 0 to indicate successful termination.
    return 0;
}
