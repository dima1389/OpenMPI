#include <stdio.h>   // For printf, scanf, fflush
#include <mpi.h>     // For MPI functions

// -----------------------------------------------------------------------------
// getInput
// -----------------------------------------------------------------------------
// Reads a single floating-point value from standard input.
// Only MPI rank 0 calls this function.
//
// Returns:
//   A double value read from the console.
// -----------------------------------------------------------------------------
double getInput()
{
    double res;
    printf("Number: ");
    fflush(stdout);              // Ensure prompt appears before input
    scanf("%lf", &res);          // Read user input
    return res;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
// Parallel summation of the arithmetic series:
//
//      S = 1 + 2 + 3 + ... + n
//
// Distributed across multiple MPI processes.
// This implementation divides the work by assigning each process a sequence:
//
//      rank starts at i = rank
//      while i <= n:
//          sum += i
//          i += number_of_processes
//
// Example:
//   If n = 10 and csize = 4,
//      rank 0 adds: 0, 4, 8
//      rank 1 adds: 1, 5, 9
//      rank 2 adds: 2, 6, ...
//      rank 3 adds: 3, 7, ...
//
// The partial sums are reduced with MPI_Reduce to obtain the final sum on rank 0.
// MPI_Wtime is used to measure performance, and the maximum elapsed time across
// processes is reported.
//
// Command-line arguments:
//   None required. Rank 0 prompts the user for the input `n`.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    double n;        // Upper limit of the sum (input)
    double sum = 0;  // Local partial sum

    int csize;       // Total number of MPI processes
    int prank;       // Rank ID of this process

    // Initialize the MPI runtime
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // ---------------------------------------------------------------------------------
    // Input stage (only rank 0 prompts the user)
    // ---------------------------------------------------------------------------------
    if (prank == 0) {
        n = getInput();
    }

    // Broadcast the input value 'n' to all processes
    MPI_Bcast(&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------------------------
    // Timing start
    // Each rank measures execution time independently.
    // ---------------------------------------------------------------------------------
    double start_time = MPI_Wtime();

    // Each rank starts at its own index:
    double i = (double)prank;
    double step = (double)csize;

    // Compute partial sum:
    // Rank r computes: r + (r + step) + (r + 2*step) + ...
    while (i <= n) {
        sum += i;
        i += step;
    }

    // ---------------------------------------------------------------------------------
    // Combine all partial sums into tsum on rank 0
    // ---------------------------------------------------------------------------------
    double tsum;  // Total sum, valid only on rank 0
    MPI_Reduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------------------------
    // Measure execution time
    // mind = maximum runtime of all ranks (worst-case time)
    // ---------------------------------------------------------------------------------
    double end_time = MPI_Wtime();
    double duration = end_time - start_time;

    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------------------------
    // Output results (rank 0 only)
    // ---------------------------------------------------------------------------------
    if (prank == 0) {
        printf("Sum of first %f integers is %f\n", n, tsum);
        printf("Elapsed time (max across processes): %f seconds\n", max_duration);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
