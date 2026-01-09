/*
 * =====================================================================================
 * MPI_Parallel_Sum_Course.c  —  Teaching Version (Heavily Commented, Beginner-First)
 * =====================================================================================
 *
 * Purpose of this program
 * -----------------------
 * This program uses MPI (Message Passing Interface) to compute the arithmetic series
 *     S = 1 + 2 + 3 + ... + n
 * in parallel by splitting the work across multiple *processes* (separate running programs).
 * It also measures runtime and reports the slowest (maximum) process time.
 *
 * High-level workflow
 * ----------------------------------
 * 1) Initialize MPI so multiple processes can cooperate (MPI_Init).
 * 2) Discover:
 *      - how many processes exist (MPI_Comm_size),
 *      - which process we are (MPI_Comm_rank).
 * 3) Input:
 *      - Only rank 0 asks the user for n (getInput).
 *      - Rank 0 broadcasts n to all processes (MPI_Bcast).
 * 4) Computation:
 *      - Each rank computes a partial sum of terms spaced by the number of processes.
 * 5) Reduction:
 *      - Add partial sums into a total sum on rank 0 (MPI_Reduce with MPI_SUM).
 *      - Collect the maximum runtime across ranks on rank 0 (MPI_Reduce with MPI_MAX).
 * 6) Output:
 *      - Only rank 0 prints the final sum and the measured time.
 * 7) Finalize MPI (MPI_Finalize).
 *
 * Build / compile instructions
 * ----------------------------
 * MPI programs must be compiled and linked against an MPI implementation.
 * The simplest and most portable method is to use the MPI wrapper compiler:
 *
 * Linux / macOS (Open MPI or MPICH):
 *   mpicc -O2 -Wall MPI_Parallel_Sum_Course.c -o MPI_Parallel_Sum_Course
 *
 * Windows (Microsoft MPI + MinGW):
 *   Microsoft MPI does not provide an mpicc wrapper for MinGW.
 *   Therefore, the compiler must be invoked directly and the MPI
 *   include and library paths must be specified explicitly.
 *
 *   Required environment variables:
 *     MSMPI_INC    → path to the Microsoft MPI include directory
 *     MSMPI_LIB64  → path to the 64-bit Microsoft MPI library directory
 *
 *   Example compilation command:
 *     gcc MPI_Parallel_Sum_Course.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Parallel_Sum_Course.exe
 *
 * Run instructions (with examples)
 * --------------------------------
 * MPI programs do NOT run as a single process if you want parallelism.
 * You start them with an MPI launcher that creates multiple processes:
 *
 *   mpiexec -n <num_processes> <program>
 *
 * Examples:
 *   mpiexec -n 4 MPI_Parallel_Sum_Course
 *   mpiexec -n 8 MPI_Parallel_Sum_Course
 *
 * Then, when prompted:
 *   Number: 10
 *
 * Expected inputs / outputs
 * -------------------------
 * Input:
 *   - One number (read as a double) entered by the user on rank 0.
 *
 * Output:
 *   - Printed only by rank 0:
 *       Sum of first <n> integers is <sum>
 *       Elapsed time (max across processes): <seconds> seconds
 *
 * Note on formatting:
 *   - The program prints n using "%f", so it will appear with decimal digits
 *     even if you enter an integer like 10.
 *
 * Common failure modes and troubleshooting tips
 * ---------------------------------------------
 * 1) Compilation error: "mpi.h: No such file or directory"
 *    - Cause: compiling with gcc/clang instead of mpicc, or MPI is not installed.
 *    - Fix: install MPI and compile with mpicc, or configure include paths.
 *
 * 2) Linker errors: "undefined reference to MPI_Init" (or similar)
 *    - Cause: MPI headers found but MPI libraries not linked.
 *    - Fix: use mpicc (recommended) or add correct MPI libraries manually.
 *
 * 3) Runtime error: "mpiexec" not found
 *    - Cause: MPI runtime not installed or not in PATH.
 *    - Fix: install the MPI runtime and ensure mpiexec is accessible.
 *
 * 4) Program “hangs” (appears stuck)
 *    - Cause in MPI generally: mismatched sends/receives or collectives called
 *      by some ranks but not others.
 *    - This program uses collectives (MPI_Bcast, MPI_Reduce) that MUST be called
 *      by ALL ranks in MPI_COMM_WORLD. If you modify the code so only some ranks
 *      call them, you can create a deadlock.
 *
 * Correctness and safety notes (pitfalls to understand)
 * -----------------------------------------------------
 * - MPI processes vs threads:
 *   - MPI uses multiple processes. Each process has its own memory.
 *   - Variables like `sum` are NOT shared between ranks.
 *
 * - Uninitialized variable risk:
 *   - Only rank 0 reads `n` from input.
 *   - Other ranks would have an uninitialized `n` if we did not broadcast it.
 *   - MPI_Bcast guarantees all ranks get the same `n` value.
 *
 * - “Works by accident” vs “guaranteed”:
 *   - If you removed MPI_Bcast, non-zero ranks might still *sometimes* have a value
 *     in `n`, but it would be garbage (undefined behavior). Any correct result would
 *     be accidental, not guaranteed by C or MPI.
 *
 * - Mathematical definition vs program’s actual behavior:
 *   - The comment says S = 1 + 2 + ... + n, but the implemented loop starts at i=rank.
 *   - This means rank 0 includes i=0 in its partial sum (0 contributes nothing).
 *     So the final result is numerically the same as summing from 1, but the program
 *     still *does* iterate over 0.
 *   - Behavior must be preserved, so we keep this exact indexing strategy.
 *
 * - Floating-point considerations:
 *   - `n` is a double, and the loop uses doubles for i/step.
 *   - If `n` is not an integer, the loop sums i values while i <= n.
 *     Example: if n = 10.7, ranks sum terms up to 10 (or 10.0) depending on rank/step.
 *   - Floating-point can introduce rounding; for very large n, adding many doubles
 *     can accumulate rounding error. This is normal for floating-point arithmetic.
 *
 * - Performance timing:
 *   - Each rank times itself. Different ranks may finish at different times.
 *   - We report the maximum duration because the overall parallel program cannot
 *     finish until the slowest rank reaches the reduction (worst-case time).
 *
 * Optional enhancements policy
 * ----------------------------
 * The prompt allows “minimal input validation” only if it does not change behavior
 * for valid inputs. This file does NOT change input handling: it still uses scanf
 * without checking the return value, because changing that could alter behavior in
 * some edge cases. We only document the risk instead of changing it.
 */

/* ------------------------------- Include files --------------------------------- */

/*
 * <stdio.h>
 * - Standard I/O library in C.
 * - Provides: printf (output), scanf (input), fflush (force output to appear).
 */
#include <stdio.h>

/*
 * <mpi.h>
 * - MPI library header.
 * - Provides MPI function declarations, types, and constants.
 */
#include <mpi.h>

/* ============================ Helper function: getInput ========================= */

/*
 * getInput()
 * ----------
 * Reads one floating-point value from the console (standard input).
 *
 * Important MPI design choice:
 * - Only rank 0 calls this function. Why?
 *   If every rank asked for input, you would see multiple prompts and the program
 *   would not know which input belongs to which rank. Typically, one rank (rank 0)
 *   handles user interaction, then shares the result with others.
 *
 * Return:
 * - A `double` value entered by the user.
 *
 * Pitfall:
 * - scanf returns the number of successfully read items.
 *   This program does not check that value (preserving original behavior).
 *   If the user enters non-numeric text, `res` may remain uninitialized, leading to
 *   undefined behavior. For teaching purposes, we note it instead of changing it.
 */
double getInput()
{
    /*
     * double res;
     * - Local variable to store the value read from the user.
     * - “Local” means it exists only inside this function.
     * - It is uninitialized until scanf writes into it.
     */
    double res;

    /*
     * printf("Number: ");
     * - Prints a prompt so the user knows what to type.
     * - Without this, the program would still read input, but the user might be confused.
     */
    printf("Number: ");

    /*
     * fflush(stdout);
     * - stdout is the output stream for normal console output.
     * - Some environments buffer output, meaning the prompt might not appear
     *   immediately (especially if it does not end with '\n').
     * - fflush forces the prompt to appear before scanf waits for input.
     *
     * If omitted:
     * - The program still works, but the prompt might appear late or not at all
     *   until after input is provided (confusing for beginners).
     */
    fflush(stdout);

    /*
     * scanf("%lf", &res);
     *
     * scanf:
     * - Reads formatted input from stdin (the console).
     *
     * "%lf":
     * - Format specifier for reading a double (“long float” in scanf terminology).
     *
     * &res:
     * - The address of res (a pointer to res).
     * - scanf needs the address to store the read value into the variable.
     *
     * If you wrote res instead of &res:
     * - scanf would interpret the numeric value in res as a memory address and
     *   attempt to write there → likely crash or corrupt memory.
     */
    scanf("%lf", &res);

    /*
     * return res;
     * - Sends the read value back to the caller.
     */
    return res;
}

/* ================================ main function ================================= */

/*
 * main(int argc, char* argv[])
 * ----------------------------
 * argc / argv:
 * - These represent command-line arguments.
 * - argc = number of arguments.
 * - argv = array of strings (char*) containing the arguments.
 *
 * Even though this program does not require command-line arguments, we still accept
 * them because MPI_Init may inspect argv for MPI runtime options (implementation detail).
 *
 * Return value:
 * - 0 means “success” to the operating system.
 */
int main(int argc, char* argv[])
{
    /* ============================ Phase 1: Variables ============================= */

    /*
     * double n;
     * - Upper limit of the sum.
     * - Rank 0 will set it from user input.
     * - Other ranks will receive it via MPI_Bcast.
     *
     * Important: At this line, n is uninitialized for all ranks.
     * It becomes valid only after input/broadcast.
     */
    double n;

    /*
     * double sum = 0;
     * - Local partial sum for this rank.
     * - Initialized to 0.0 so we start from a known value.
     *
     * MPI concept reminder:
     * - Each rank has its own independent `sum` variable in its own process memory.
     * - There is no shared variable `sum` across processes.
     */
    double sum = 0;

    /*
     * int csize;
     * - Total number of processes in MPI_COMM_WORLD.
     * - Set by MPI_Comm_size.
     */
    int csize;

    /*
     * int prank;
     * - Rank (“process ID”) of this process inside MPI_COMM_WORLD.
     * - Set by MPI_Comm_rank.
     */
    int prank;

    /* ============================ Phase 2: MPI setup ============================= */

    /*
     * MPI_Init(&argc, &argv);
     *
     * Must be called before almost any other MPI call.
     *
     * Why pass &argc and &argv?
     * - MPI implementations may remove MPI-specific arguments from argv.
     * - Passing pointers allows MPI_Init to modify argc/argv if needed.
     *
     * If omitted:
     * - Calling MPI_Comm_size / MPI_Comm_rank / MPI_Bcast / MPI_Reduce is invalid.
     */
    MPI_Init(&argc, &argv);

    /*
     * MPI_Comm_size(MPI_COMM_WORLD, &csize);
     * - Writes the number of ranks into csize.
     * - &csize is a pointer to csize so MPI can store the result.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    /*
     * MPI_Comm_rank(MPI_COMM_WORLD, &prank);
     * - Writes this rank’s ID into prank (0..csize-1).
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /* ============================ Phase 3: Input ================================ */

    /*
     * if (prank == 0) { n = getInput(); }
     *
     * Only rank 0 interacts with the user.
     * All other ranks skip this block.
     *
     * Why this matters:
     * - Avoids multiple processes printing "Number:" at the same time.
     * - Prevents confusion and inconsistent input handling.
     */
    if (prank == 0) {
        /*
         * n = getInput();
         * - Calls getInput and stores the returned double into n.
         * - After this line, only rank 0 has a valid n.
         */
        n = getInput();
    }

    /*
     * MPI_Bcast(&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     *
     * MPI_Bcast = broadcast:
     * - One rank (the “root”) sends the same data to every rank in the communicator.
     *
     * Parameters:
     * - &n: address of the variable to send/receive.
     * - 1: number of elements.
     * - MPI_DOUBLE: datatype is double.
     * - 0: root rank (rank 0 is the sender; all others are receivers).
     * - MPI_COMM_WORLD: communicator containing all ranks.
     *
     * Critical rule for collectives:
     * - EVERY rank in MPI_COMM_WORLD must call MPI_Bcast here.
     * - If some ranks call it and others do not, the program can deadlock.
     *
     * After this call:
     * - All ranks have the same valid value of n.
     */
    MPI_Bcast(&n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ============================ Phase 4: Timing start ========================= */

    /*
     * double start_time = MPI_Wtime();
     *
     * MPI_Wtime:
     * - Returns a floating-point time value (in seconds) from an MPI-defined clock.
     * - Used for measuring elapsed time.
     *
     * Important:
     * - Each rank calls MPI_Wtime independently, so start times are not necessarily
     *   identical across ranks.
     * - That is fine: we measure each rank’s local runtime and then reduce with MPI_MAX.
     */
    double start_time = MPI_Wtime();

    /* ============================ Phase 5: Computation ========================== */

    /*
     * double i = (double)prank;
     *
     * i is the current term this rank will add.
     *
     * Why start at prank?
     * - This is a simple “cyclic distribution” pattern:
     *     rank r computes r, r+csize, r+2*csize, ...
     *
     * Note about series definition:
     * - The mathematical series is usually 1..n.
     * - This code includes the term 0 from rank 0, but adding 0 does not change the sum.
     * - We keep this exact behavior to preserve the program’s results.
     */
    double i = (double)prank;

    /*
     * double step = (double)csize;
     *
     * step is how far we jump each loop iteration.
     * If there are P processes, each rank takes every P-th number.
     */
    double step = (double)csize;

    /*
     * while (i <= n) { sum += i; i += step; }
     *
     * Control structure: while-loop
     * - Repeats as long as the condition (i <= n) is true.
     *
     * What it computes:
     * - Rank r sums: r + (r+P) + (r+2P) + ... up to n.
     *
     * Example (n=10, P=4):
     * - rank 0: 0, 4, 8
     * - rank 1: 1, 5, 9
     * - rank 2: 2, 6, 10
     * - rank 3: 3, 7
     *
     * Correctness note:
     * - This covers each integer from 0..n exactly once (for integer n >= 0),
     *   distributed across ranks, so the reduction gives the total sum.
     *
     * Floating-point note:
     * - i and n are doubles. Comparisons like i <= n can be affected by rounding
     *   for some values, but for integer-like values this is typically fine.
     */
    while (i <= n) {
        /*
         * sum += i;
         * - Adds the current term i to this rank’s local sum.
         */
        sum += i;

        /*
         * i += step;
         * - Move to the next term assigned to this rank.
         * - Without this, i would never change and the loop could be infinite.
         */
        i += step;
    }

    /* ============================ Phase 6: Reduction (sum) ====================== */

    /*
     * double tsum;
     * - Variable intended to hold the total sum on rank 0.
     *
     * Important:
     * - On non-zero ranks, tsum is not initialized by the program in a meaningful way.
     * - MPI_Reduce only stores the reduced result in the receive buffer on the root rank.
     * - Therefore, tsum should only be used/printed on rank 0.
     */
    double tsum;

    /*
     * MPI_Reduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     *
     * MPI_Reduce:
     * - Combines values from all ranks into one result on the root rank.
     *
     * Parameters:
     * - &sum: address of local contribution (each rank’s partial sum).
     * - &tsum: address where the root rank stores the final result.
     * - 1: one element.
     * - MPI_DOUBLE: each element is a double.
     * - MPI_SUM: operation is addition (sum all partial sums).
     * - 0: root rank (the rank that receives the final result).
     * - MPI_COMM_WORLD: communicator.
     *
     * Deadlock rule:
     * - Like all collectives, every rank must call MPI_Reduce here.
     */
    MPI_Reduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ============================ Phase 7: Timing end =========================== */

    /*
     * double end_time = MPI_Wtime();
     * - Capture current time after computation and reduction call above.
     */
    double end_time = MPI_Wtime();

    /*
     * double duration = end_time - start_time;
     * - Local elapsed time for this rank, in seconds.
     */
    double duration = end_time - start_time;

    /*
     * double max_duration;
     * - Will hold the maximum duration across all ranks, but only valid on rank 0
     *   after the reduction below.
     */
    double max_duration;

    /*
     * MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     *
     * This reduction uses MPI_MAX:
     * - It selects the largest duration among all ranks.
     *
     * Why maximum?
     * - In parallel programs, the slowest rank often determines overall completion time,
     *   because other ranks may need to wait at synchronization points or collectives.
     */
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ============================ Phase 8: Output =============================== */

    /*
     * if (prank == 0) { ... }
     *
     * Only rank 0 prints:
     * - Avoids multiple ranks printing simultaneously, which can interleave output.
     * - Rank 0 is the only rank guaranteed to have valid values in `tsum` and `max_duration`.
     */
    if (prank == 0) {
        /*
         * printf("Sum of first %f integers is %f\n", n, tsum);
         *
         * Note: %f prints doubles as floating-point (e.g., "10.000000").
         * The text says "integers" even though n is a double; we preserve the exact output
         * format and wording to preserve behavior.
         */
        printf("Sum of first %f integers is %f\n", n, tsum);

        /*
         * printf("Elapsed time (max across processes): %f seconds\n", max_duration);
         * - Prints the maximum runtime across ranks (a conservative timing).
         */
        printf("Elapsed time (max across processes): %f seconds\n", max_duration);
    }

    /* ============================ Phase 9: Cleanup ============================== */

    /*
     * MPI_Finalize();
     * - Cleanly shuts down the MPI environment for this process.
     * - No MPI calls are allowed after this point.
     */
    MPI_Finalize();

    /*
     * return 0;
     * - Indicate successful termination to the operating system.
     */
    return 0;
}
