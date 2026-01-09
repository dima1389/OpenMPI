/*
 * File:
 *   mpi_allreduce.c
 *
 * Purpose:
 *   Demonstrate MPI_Allreduce as a collective global reduction that returns the result to all ranks.
 *
 * Description:
 *   This example demonstrates collective reduction with MPI_Allreduce under the MPI_COMM_WORLD communicator.
 *   Observable outcome: every rank computes identical global aggregates (sum, max, average) without a separate broadcast.
 *
 * Key concepts:
 *   - collectives
 *   - deterministic (for associative/commutative ops like SUM/MAX on identical inputs and rank participation)
 *   - performance effect: latency (collective synchronization cost per call)
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI environment and parse optional iteration count
 *   2) Compute global sum/max of one local value per rank using MPI_Allreduce
 *   3) Time repeated MPI_Allreduce calls to illustrate collective cost
 *   4) Report results (rank 0 prints, but all ranks hold the same reduced values)
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Allreduce
 *   - MPI_Barrier
 *   - MPI_Wtime
 *   - MPI_Abort
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_allreduce.c -o mpi_allreduce
 *
 * Execution:
 *   mpiexec -n <P> mpi_allreduce [iterations]
 *
 * Inputs:
 *   - Command-line arguments: iterations (optional, positive integer; default: 100000)
 *   - Interactive input: none
 *
 * References:
 *   - MPI Standard: Collective Communication / MPI_Allreduce
 */

#include <mpi.h>      // Provides MPI_Init, MPI_Allreduce, MPI_Wtime, and all MPI datatypes and collectives.
#include <stdio.h>    // Provides printf and fprintf for user-facing output and error reporting.
#include <stdlib.h>   // Provides strtoll and exit-related facilities for robust argument parsing.
#include <errno.h>    // Provides errno to detect strtoll conversion errors precisely.
#include <limits.h>   // Provides LLONG_MAX for overflow checking during integer parsing.

static long long parse_positive_ll_or_abort(int argc, char *argv[], int rank) // Defines a helper to parse a positive integer argument consistently across ranks.
{
    long long iterations = 100000; // Sets a conservative default iteration count to make timing observable without being excessively slow.

    if (argc >= 2) { // Checks whether the optional iterations argument is present.
        char *end = NULL; // Declares an end-pointer used by strtoll to validate full-string numeric conversion.
        errno = 0; // Resets errno so strtoll error detection is unambiguous for this conversion.
        long long tmp = strtoll(argv[1], &end, 10); // Converts argv[1] to a base-10 signed integer while capturing the first invalid character.
        if (errno != 0) { // Detects conversion errors such as ERANGE (overflow/underflow) signaled via errno.
            if (rank == 0) { // Restricts user-facing error messages to rank 0 to avoid duplicated output.
                fprintf(stderr, "ERROR: Failed to parse iterations (errno=%d).\n", errno); // Reports the parsing failure with diagnostic errno.
            }
            MPI_Abort(MPI_COMM_WORLD, 1); // Terminates all ranks because inconsistent input handling would break collective participation.
        }
        if (end == argv[1] || *end != '\0') { // Validates that at least one digit was parsed and that no trailing non-numeric characters remain.
            if (rank == 0) { // Ensures only rank 0 prints the usage message for clarity.
                fprintf(stderr, "Usage: %s [iterations]\n", argv[0]); // Prints correct program usage to standard error.
            }
            MPI_Abort(MPI_COMM_WORLD, 1); // Aborts the MPI job because ranks must agree on the loop bound for meaningful timing.
        }
        if (tmp <= 0 || tmp == LLONG_MAX) { // Enforces a strictly positive iteration count and rejects obvious overflow sentinel values.
            if (rank == 0) { // Limits the error message to rank 0 to prevent repeated output from all ranks.
                fprintf(stderr, "ERROR: iterations must be a positive integer.\n"); // Explains the input constraint to the user.
            }
            MPI_Abort(MPI_COMM_WORLD, 1); // Aborts the full communicator to avoid undefined behavior from invalid loop bounds.
        }
        iterations = tmp; // Commits the validated argument value as the iteration count used by all ranks.
    }

    return iterations; // Returns the agreed-upon iteration count to the caller.
}

int main(int argc, char *argv[]) // Declares the program entry point; MPI programs typically initialize MPI within main.
{
    int rank = -1; // Initializes rank with a sentinel to make misuse before MPI_Comm_rank easier to spot during debugging.
    int size = 0; // Initializes size to a safe value to avoid accidental division by zero before MPI_Comm_size is called.

    MPI_Init(&argc, &argv); // Initializes the MPI runtime; required before most MPI calls and collective operations.

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Queries the calling process' rank ID within MPI_COMM_WORLD for role decisions and diagnostics.
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Queries the total number of ranks in MPI_COMM_WORLD to compute global averages and validate scale.

    long long iterations = parse_positive_ll_or_abort(argc, argv, rank); // Parses the optional iterations argument consistently and aborts on invalid input.

    const double local_value = (double)(rank + 1); // Defines a simple deterministic per-rank contribution (1..size) to make reductions easy to verify.

    double global_sum = 0.0; // Declares storage for the SUM reduction result that will be returned to every rank.
    double global_max = 0.0; // Declares storage for the MAX reduction result that will be returned to every rank.

    MPI_Allreduce(&local_value, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Computes the global sum of local_value across all ranks and distributes it back to all ranks.
    MPI_Allreduce(&local_value, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Computes the global maximum of local_value across all ranks and distributes it back to all ranks.

    const double global_avg = global_sum / (double)size; // Computes the global average locally, using the globally reduced sum and communicator size.

    if (rank == 0) { // Uses rank 0 as the designated reporting rank to avoid redundant output from all processes.
        printf("MPI_Allreduce demo (P=%d)\n", size); // Prints the communicator size to contextualize the reduction results.
        printf("  local_value per rank: rank+1\n"); // Describes the deterministic per-rank input function used for reductions.
        printf("  global_sum = %.0f (expected %.0f)\n", global_sum, (double)size * (double)(size + 1) / 2.0); // Prints the sum and an analytically expected reference value.
        printf("  global_max = %.0f (expected %.0f)\n", global_max, (double)size); // Prints the max and its expected value (size) given local_value = rank+1.
        printf("  global_avg = %.6f\n", global_avg); // Prints the average to show derived global statistics available on every rank.
        printf("  iterations for timing = %lld\n", iterations); // Prints the iteration count used in the micro-benchmark loop for transparency and reproducibility.
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronizes ranks so the timing window starts from a consistent point across the communicator.

    double t0 = MPI_Wtime(); // Captures the start time using MPI's wall-clock timer for portable timing across platforms.
    double accum = local_value; // Initializes an accumulator to provide a changing input value and prevent trivial compiler optimizations.
    for (long long i = 0; i < iterations; ++i) { // Repeats the collective to amortize timer resolution and expose steady-state latency.
        double tmp = 0.0; // Declares a temporary reduction target that receives the all-reduced value each iteration.
        MPI_Allreduce(&accum, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // Performs an all-reduce each iteration to benchmark collective cost under repeated use.
        accum = tmp * 1e-300 + local_value; // Perturbs the next input slightly while keeping values finite, discouraging dead-code elimination.
    }
    double t1 = MPI_Wtime(); // Captures the stop time after the loop to compute elapsed time for the repeated collective operations.

    double local_elapsed = t1 - t0; // Computes the elapsed time on this rank for the timed section.
    double max_elapsed = 0.0; // Declares storage for the worst-case (maximum) elapsed time across ranks.
    MPI_Allreduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // Computes the maximum elapsed time so rank 0 can report an overall SPMD wall-time metric.

    if (rank == 0) { // Restricts timing output to rank 0 to keep output concise while still reporting the relevant global metric.
        double per_call = max_elapsed / (double)iterations; // Computes average time per MPI_Allreduce call using the max elapsed time as the conservative bound.
        printf("Timing (max across ranks): total = %.6f s, per Allreduce = %.3e s\n", max_elapsed, per_call); // Reports total time and per-call cost to characterize collective latency.
    }

    if (accum == -1.0) { // Adds an impossible branch to reference accum in a way that is semantically inert but prevents aggressive optimization assumptions.
        printf("unreachable: %.6f\n", accum); // Provides a side-effect in the impossible branch so accum remains observable to the compiler.
    }

    MPI_Finalize(); // Finalizes the MPI runtime and releases MPI resources; required for a clean MPI program shutdown.

    return 0; // Returns a success exit code to the host environment to indicate normal termination.
}
