/*
 * File:
 *   mpi_alltoall_two_digit_exchange.c
 *
 * Purpose:
 *   Demonstrate MPI_Alltoall() as a collective all-to-all personalized exchange.
 *
 * Description:
 *   This example demonstrates an all-to-all communication pattern where each rank sends one
 *   integer to every other rank using MPI_Alltoall(). Each sent integer is a two-digit number XY:
 *     - X (tens digit) is the sender rank (must be a single decimal digit 0..9)
 *     - Y (ones digit) is a per-destination random digit 0..9
 *   Observable outcome: each rank prints the full set of received two-digit values, one from each
 *   other rank, in a deterministic print order (rank-by-rank) while the payload is nondeterministic
 *   due to random digits.
 *
 * Key concepts:
 *   - ranks, communicators, collectives (all-to-all), barriers, abort semantics
 *   - deterministic output ordering vs nondeterministic message payload generation
 *   - performance effect: all-to-all bandwidth/latency scaling (O(P^2) message relationships)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and query rank/size
 *   2) Validate that size <= 10 to preserve the XY encoding constraint
 *   3) Allocate per-rank send/receive buffers sized by communicator size
 *   4) Populate sendbuf[dest] with XY values (rank as tens digit, random as ones digit)
 *   5) Exchange values using MPI_Alltoall (1 int to/from every rank)
 *   6) Print received values in rank order using MPI_Barrier to avoid interleaving
 *   7) Release resources and finalize MPI
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Alltoall, MPI_Barrier, MPI_Abort
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_alltoall_two_digit_exchange.c -o mpi_alltoall_two_digit_exchange
 *
 * Execution:
 *   mpiexec -n <P> ./mpi_alltoall_two_digit_exchange
 *
 * Inputs:
 *   - Command-line arguments: none
 *   - Interactive input: none
 *
 * References:
 *   - MPI Standard: Collective Communication (MPI_Alltoall)
 *   - MPI Standard: Process termination (MPI_Abort)
 */

#include <stdio.h>   // Declares printf(), fprintf(), and stderr for formatted output and error reporting.
#include <stdlib.h>  // Declares malloc(), free(), rand(), rand_r(), and srand() for dynamic allocation and RNG.
#include <time.h>    // Declares time() for obtaining a time-based seed value.
#include <mpi.h>     // Declares the MPI API used for initialization, collectives, synchronization, and abort.

/*
 * Returns a pseudo-random decimal digit in the range [0, 9].
 * The implementation selects a platform-appropriate RNG interface.
 */
static int random_digit(unsigned int *seed)                 // Defines a helper function to generate one decimal digit using an RNG state.
{
#if defined(_WIN32)                                         // Selects the Windows-specific branch where rand_r() is typically unavailable.
    (void)seed;                                             // Explicitly marks the unused seed parameter to avoid compiler warnings on Windows.
    return rand() % 10;                                     // Produces a digit by reducing rand() output modulo 10 (global RNG state).
#else                                                       // Selects the POSIX-like branch where rand_r() may be available.
    return (int)(rand_r(seed) % 10);                        // Produces a digit via rand_r() using a per-call seed to reduce shared-state races.
#endif                                                      // Ends the platform selection conditional.
}

int main(int argc, char *argv[])                            // Defines the MPI program entry point with standard command-line parameters.
{
    int rank, size;                                         // Declares the calling process rank and the communicator size for MPI_COMM_WORLD.

    MPI_Init(&argc, &argv);                                 // Initializes the MPI runtime and allows MPI to process implementation-specific argv options.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                   // Queries the rank ID of this process within the global communicator.
    MPI_Comm_size(MPI_COMM_WORLD, &size);                   // Queries the total number of processes participating in the global communicator.

    /* Enforce "rank is the first digit" => rank must be 0..9 => size <= 10 */
    if (size > 10) {                                        // Validates the strict XY encoding constraint by requiring all ranks fit in one decimal digit.
        if (rank == 0) {                                    // Restricts the error message emission to rank 0 to avoid duplicate identical diagnostics.
            fprintf(stderr,                                 // Writes a diagnostic to standard error to distinguish it from normal program output.
                    "ERROR: This task requires size <= 10 so that each rank fits into one decimal digit.\n"
                    "You started %d processes.\n",
                    size);                                  // Prints the observed communicator size that violated the constraint.
        }
        MPI_Abort(MPI_COMM_WORLD, 1);                       // Terminates all ranks in the communicator with a nonzero error code for the launcher.
        /* Not reached */                                   // Documents that control flow does not continue past MPI_Abort() in a conforming implementation.
    }

#if defined(_WIN32)                                         // Selects Windows-specific RNG seeding because rand() uses global process state.
    /* rand() is global state; seed per rank */
    srand((unsigned)time(NULL) ^ (unsigned)(rank * 2654435761u)); // Seeds rand() with a time-based value mixed with rank to reduce identical streams.
#endif                                                      // Ends the Windows-specific RNG seeding block.

    int *sendbuf = (int *)malloc((size_t)size * sizeof(int)); // Allocates the per-destination send buffer with one int slot per rank in the communicator.
    int *recvbuf = (int *)malloc((size_t)size * sizeof(int)); // Allocates the per-source receive buffer with one int slot per rank in the communicator.
    if (!sendbuf || !recvbuf) {                               // Checks for allocation failure because collective communication requires valid buffers.
        fprintf(stderr, "Rank %d: malloc failed\n", rank);    // Reports which rank failed to allocate memory to aid debugging in distributed runs.
        MPI_Abort(MPI_COMM_WORLD, 2);                         // Aborts the full job because continuing would dereference null pointers.
    }

    /* Different RNG stream per rank (and per run) */
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(rank * 1103515245u); // Initializes a per-rank seed by mixing time and rank.

    /* Prepare one message per destination */
    for (int dest = 0; dest < size; ++dest) {               // Iterates over every destination rank to populate the send buffer for all-to-all exchange.
        if (dest == rank) {                                 // Detects the self-destination slot which is not meaningfully used for this task.
            /* Self-message not used; keep placeholder */
            sendbuf[dest] = -1;                             // Stores a sentinel value to keep the buffer fully defined for MPI_Alltoall.
        } else {                                            // Handles the general case where the destination is a different rank.
            int tens = rank;                                // Sets the tens digit to the sender rank to encode the required XY format.
            int ones = random_digit(&seed);                 // Generates a destination-specific ones digit using the per-rank seed state.
            sendbuf[dest] = tens * 10 + ones;               // Encodes XY as a two-digit integer and stores it in the destination slot.
        }
    }

    /* Exchange: recvbuf[src] is what we got from process 'src' */
    MPI_Alltoall(sendbuf, 1, MPI_INT,                       // Sends one MPI_INT from each sendbuf[dest] slot to the matching destination rank.
                 recvbuf, 1, MPI_INT,                       // Receives one MPI_INT from each source rank into recvbuf[src] in rank order.
                 MPI_COMM_WORLD);                           // Performs the collective exchange over the global communicator.

    /* Print in rank order to avoid interleaving */
    for (int r = 0; r < size; ++r) {                        // Iterates over ranks to serialize output so stdout lines do not interleave across processes.
        MPI_Barrier(MPI_COMM_WORLD);                        // Synchronizes all ranks so only one designated rank prints at a time per iteration.
        if (r == rank) {                                    // Selects exactly one rank (the current iteration's rank) to print its receive results.
            printf("Process %d received:", rank);           // Prints a rank-identified prefix to associate the following values with the printing process.
            for (int src = 0; src < size; ++src) {          // Iterates over all source ranks to print the value received from each source.
                if (src == rank) continue;                  // Skips the self-source entry because the self-message is explicitly treated as unused.
                printf(" %d", recvbuf[src]);                // Prints the two-digit value received from the given source rank.
            }
            printf("\n");                                   // Terminates the output line to keep each rank's report on a single line.
            fflush(stdout);                                 // Flushes stdout to force timely emission under buffered I/O and MPI launcher redirection.
        }
    }

    free(sendbuf);                                          // Releases dynamically allocated send buffer to avoid memory leaks on long-running processes.
    free(recvbuf);                                          // Releases dynamically allocated receive buffer to avoid memory leaks on long-running processes.

    MPI_Finalize();                                         // Finalizes the MPI runtime and releases MPI-internal resources before process exit.
    return 0;                                               // Returns a success status code to the hosting environment and MPI launcher.
}
