/*
 * File:
 *   mpi_sum_1_to_n_reduce.c
 *
 * Purpose:
 *   Demonstrate MPI_Reduce for computing a global sum from per-rank partial sums.
 *
 * Description:
 *   This example computes Sum(1..N) in parallel by distributing the integer range [1, N]
 *   across MPI ranks using an uneven block decomposition (handles N not divisible by P).
 *   Each rank computes its local arithmetic-series sum and rank 0 aggregates the result
 *   using MPI_Reduce with MPI_SUM.
 *   Observable outcome: rank 0 prints the correct global sum for the provided N.
 *
 * Key concepts:
 *   - ranks, communicator (MPI_COMM_WORLD), collective communication
 *   - broadcast (problem size distribution), reduction (global aggregation)
 *   - load imbalance avoidance via remainder-aware block partitioning
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and obtain rank/size
 *   2) Read N on rank 0 and broadcast N to all ranks
 *   3) Compute each rank's sub-interval of [1, N]
 *   4) Compute local sum and reduce to rank 0, then print
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Bcast, MPI_Reduce, MPI_Abort
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_sum_1_to_n_reduce.c -o mpi_sum_1_to_n_reduce
 *
 * Execution:
 *   mpiexec -n <P> ./mpi_sum_1_to_n_reduce <N>
 *   mpiexec -n <P> ./mpi_sum_1_to_n_reduce        (interactive input on rank 0)
 *
 * Inputs:
 *   - Command-line arguments: N (non-negative integer), optional
 *   - Interactive input: N read on rank 0 if not provided on the command line
 *
 * References:
 *   - MPI Standard: Collective Communication (MPI_Bcast, MPI_Reduce)
 */

#include <stdio.h>   // Provides printf(), fprintf(), scanf(), fflush(), stderr.
#include <stdlib.h>  // Provides strtoll() for robust command-line parsing.
#include <mpi.h>     // Provides MPI_Init(), MPI_Bcast(), MPI_Reduce(), MPI_Finalize(), MPI types.

int main(int argc, char *argv[])                       // Defines program entry point with MPI-style argc/argv.
{
    int rank = 0;                                      // Stores this process rank within MPI_COMM_WORLD for role selection.
    int size = 0;                                      // Stores the communicator size to parameterize the decomposition.
    long long N = 0;                                   // Stores the global upper bound N for Sum(1..N) as 64-bit integer.

    MPI_Init(&argc, &argv);                            // Initializes the MPI runtime before any other MPI calls.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);              // Retrieves the calling process rank in the world communicator.
    MPI_Comm_size(MPI_COMM_WORLD, &size);              // Retrieves the total number of ranks participating in the world communicator.

    /* Input: either command line or interactive (rank 0 only). */
    if (rank == 0) {                                   // Restricts user interaction and parsing responsibility to rank 0.
        if (argc >= 2) {                               // Checks for an optional command-line argument providing N.
            char *end = NULL;                          // Receives the end-pointer from strtoll() to validate full-string parse.
            long long tmp = strtoll(argv[1], &end, 10); // Converts argv[1] to base-10 long long while tracking parse validity.
            if (end == argv[1] || *end != '\0' || tmp < 0) { // Rejects empty parse, trailing junk, or negative N values.
                fprintf(stderr, "Usage: %s <N>  (N must be a non-negative integer)\n", argv[0]); // Reports correct usage and constraint.
                MPI_Abort(MPI_COMM_WORLD, 1);          // Aborts all ranks to avoid deadlock from inconsistent control flow.
            }
            N = tmp;                                   // Commits validated N to the shared problem-size variable.
        } else {                                       // Falls back to interactive input when N is not provided.
            printf("Enter N (non-negative integer): "); // Prompts the user for N on standard output.
            fflush(stdout);                            // Flushes stdout to ensure the prompt is visible before blocking on input.
            if (scanf("%lld", &N) != 1 || N < 0) {      // Validates that one long long was read and that it is non-negative.
                fprintf(stderr, "Invalid input. N must be a non-negative integer.\n"); // Reports invalid input without proceeding.
                MPI_Abort(MPI_COMM_WORLD, 1);          // Aborts all ranks to prevent collective mismatch on subsequent calls.
            }
        }
    }

    /* Broadcast N so every rank knows the problem size. */
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD); // Distributes N from root (rank 0) to all ranks for consistent computation.

    /*
     * Compute each rank's block [local_start, local_end] within [1, N].
     *
     * Let:
     *   q = N / size           (base block size)
     *   r = N % size           (remainder)
     *
     * Ranks 0..(r-1) get (q+1) elements, remaining ranks get q elements.
     */
    long long q = (size > 0) ? (N / size) : 0;         // Computes the base block size while guarding against invalid size values.
    long long r = (size > 0) ? (N % size) : 0;         // Computes the remainder to assign one extra element to the first r ranks.

    long long local_count = (rank < r) ? (q + 1) : q;  // Assigns per-rank element count for an even-as-possible partition.

    /* Number of elements assigned to ranks smaller than me (prefix sum). */
    long long prefix = rank * q + (rank < r ? rank : r); // Computes the global offset by counting all elements on lower ranks.

    long long local_start = 1 + prefix;                // Computes the inclusive start index of this rank's interval in [1, N].
    long long local_end   = local_start + local_count - 1; // Computes the inclusive end index consistent with local_count.

    /* Local sum using arithmetic series formula on the local interval. */
    long long local_sum = 0;                            // Initializes the per-rank partial sum accumulator.
    if (local_count > 0) {                              // Skips computation for ranks assigned an empty interval (e.g., N < size).
        long long a = local_start;                      // Names the first term of the local arithmetic series for clarity.
        long long b = local_end;                        // Names the last term of the local arithmetic series for clarity.
        long long cnt = local_count;                    // Names the number of terms to keep the formula explicit and readable.
        local_sum = (a + b) * cnt / 2;                  // Applies sum_{k=a..b} k = (a+b)*cnt/2 for O(1) local computation.
    }

    long long global_sum = 0;                           // Stores the final reduced sum on rank 0 (undefined on non-root ranks).
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // Aggregates all local sums to rank 0 via MPI_SUM.

    if (rank == 0) {                                    // Restricts output to rank 0 to avoid duplicated prints.
        printf("Sum(1..%lld) = %lld\n", N, global_sum); // Prints the computed global sum for the provided N.
    }

    MPI_Finalize();                                     // Shuts down the MPI runtime and releases MPI resources cleanly.
    return 0;                                           // Returns success status to the operating system.
}
