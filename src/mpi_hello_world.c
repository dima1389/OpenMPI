/*
 * File:
 *   mpi_point_to_point_greetings.c
 *
 * Purpose:
 *   Demonstrate basic MPI point-to-point communication with rank-based control flow.
 *
 * Description:
 *   This example demonstrates blocking point-to-point messaging (MPI_Send/MPI_Recv)
 *   in MPI_COMM_WORLD, where all non-root ranks send a null-terminated greeting
 *   string to rank 0, and rank 0 receives and prints one message from each rank.
 *   Observable outcome: rank 0 prints its own greeting plus one greeting per sender rank.
 *
 * Key concepts:
 *   - ranks, communicator (MPI_COMM_WORLD), point-to-point communication
 *   - deterministic blocking behavior (explicit source ranks in MPI_Recv)
 *   - performance note: serialized receives at the root (potential root bottleneck)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and query rank/size
 *   2) Non-root ranks format and send a greeting to rank 0
 *   3) Rank 0 prints its own greeting and receives one message from each rank
 *   4) Finalize MPI
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize
 *   - MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Send, MPI_Recv
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_point_to_point_greetings.c -o mpi_point_to_point_greetings
 *
 * Execution:
 *   mpiexec -n <P> mpi_point_to_point_greetings
 *
 * Inputs:
 *   - Command-line arguments: none
 *   - Interactive input: none
 *
 * References:
 *   - MPI Standard: Point-to-point communication (MPI_Send, MPI_Recv)
 */

#include <stdio.h>      // Provides printf/sprintf for formatted output and string formatting.
#include <string.h>     // Provides strlen to compute the payload length of the C string.
#include <mpi.h>        // Provides MPI APIs and types (e.g., MPI_Init, MPI_Send, MPI_COMM_WORLD).

const int MAX_STRING = 100; // Defines the fixed receive/send buffer capacity (including the '\0').

int main(void) // Entry point; uses no command-line arguments, so MPI_Init is called with NULLs.
{
    char gret[MAX_STRING];  // Allocates a stack buffer for sending/receiving a null-terminated greeting.
    int csize;              // Stores the total number of ranks in MPI_COMM_WORLD.
    int prank;              // Stores the calling rank ID within MPI_COMM_WORLD.

    MPI_Init(NULL, NULL); // Initializes the MPI runtime; required before any other MPI call.

    MPI_Comm_size(MPI_COMM_WORLD, &csize); // Queries communicator size so ranks can iterate over all participants.

    MPI_Comm_rank(MPI_COMM_WORLD, &prank); // Queries this process's rank to select sender vs root behavior.

    if (prank != 0) { // Selects all non-root ranks to act as senders and avoid root self-send complexity.

        sprintf(gret, "Greets from process %d of %d!", prank, csize); // Builds a rank-annotated greeting string in the local buffer.

        MPI_Send(gret, (int)(strlen(gret) + 1), MPI_CHAR, 0, 0, MPI_COMM_WORLD); // Sends the null-terminated string to rank 0 using a matching tag.

    } else { // Selects rank 0 as the designated root that collects and prints all greetings.

        printf("Greets from process %d of %d!\n", prank, csize); // Prints rank 0's greeting locally to include the root in the output set.

        for (int q = 1; q < csize; q++) { // Iterates over all sender ranks to receive exactly one message from each.

            MPI_Recv(gret, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receives from a specific source rank to ensure deterministic pairing.

            printf("%s\n", gret); // Prints the received C string, relying on the sender transmitting the terminating '\0'.
        }
    }

    MPI_Finalize(); // Finalizes MPI and releases runtime resources; no MPI calls are valid after this point.

    return 0; // Returns success status to the host environment.
}
