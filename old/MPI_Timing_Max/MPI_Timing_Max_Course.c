/*
 * =====================================================================================
 * MPI_Timing_Max_Course.c  —  Teaching Version (Heavily Commented, Beginner-First)
 * =====================================================================================
 *
 * Purpose (1–3 sentences)
 * -----------------------
 * This program demonstrates how to measure elapsed (wall-clock) execution time on each
 * MPI process, how to synchronize all processes before timing using MPI_Barrier, and how
 * to compute the maximum elapsed time across all processes using MPI_Reduce with MPI_MAX.
 *
 * High-level workflow (bullet steps)
 * ----------------------------------
 * 1) Initialize MPI (MPI_Init) so multiple processes can run the same program together.
 * 2) Discover:
 *      - this process’s rank (MPI_Comm_rank),
 *      - total number of processes (MPI_Comm_size).
 * 3) Synchronize all processes at a barrier (MPI_Barrier) so timing starts from the same
 *    logical point in the code.
 * 4) Start a local timer (MPI_Wtime).
 * 5) Run a “simulated workload” whose runtime depends on rank (intentional load imbalance).
 * 6) Stop the local timer and compute local elapsed time.
 * 7) Print each process’s local elapsed time (output order may be interleaved).
 * 8) Reduce all local times to the maximum time on rank 0 (MPI_Reduce + MPI_MAX).
 * 9) Rank 0 prints the maximum time, which typically represents the effective parallel runtime.
 * 10) Finalize MPI (MPI_Finalize).
 *
 * Key MPI concepts introduced
 * ---------------------------
 * - MPI process:
 *   An OS process running the same executable. MPI is typically used for multi-process
 *   parallelism (not threads). Each process has its own private memory.
 *
 * - Rank:
 *   An integer ID assigned to each process within a communicator (here MPI_COMM_WORLD).
 *   Ranks range from 0 to size-1.
 *
 * - Communicator (MPI_COMM_WORLD):
 *   A group of processes that can communicate. MPI_COMM_WORLD contains all processes
 *   launched together by mpiexec.
 *
 * - Collective operations:
 *   MPI_Barrier and MPI_Reduce are “collectives”, meaning ALL ranks in the communicator
 *   must call them, otherwise the program can deadlock.
 *
 * Why “maximum time” is usually the relevant metric
 * -------------------------------------------------
 * Many MPI programs follow the SPMD model (Single Program, Multiple Data):
 * - All ranks execute the same code structure, but operate on different data.
 * - The overall program typically cannot finish a phase until the slowest rank finishes,
 *   because later steps often involve synchronization or communication.
 *
 * Therefore, the maximum elapsed time (slowest rank) is often the effective runtime.
 *
 * Build / compile instructions
 * ----------------------------
 * Use an MPI wrapper compiler so the correct MPI headers and libraries are used:
 *
 * Linux / macOS (Open MPI or MPICH):
 *   mpicc -O2 -Wall MPI_Timing_Max_Course.c -o MPI_Timing_Max_Course
 *
 * Windows (varies by MPI distribution):
 *   - If you have mpicc, use it similarly.
 *   - Otherwise you must provide the include/lib paths and link against your MPI libraries.
 *
 * Run instructions (with examples)
 * --------------------------------
 * MPI programs are launched with an MPI launcher that starts multiple processes:
 *
 *   mpiexec -n <num_processes> <program>
 *
 * Examples:
 *   mpiexec -n 4 ./MPI_Timing_Max_Course
 *   mpiexec -n 8 ./MPI_Timing_Max_Course
 *
 * Expected inputs / outputs
 * -------------------------
 * Inputs:
 *   - None.
 *
 * Outputs:
 *   - Each process prints its local elapsed time:
 *       Process r: local elapsed time = ...
 *   - Rank 0 prints the maximum elapsed time across all processes.
 *
 * Important note about output order:
 * - The prints from different processes are not synchronized; lines may appear interleaved
 *   or in varying order across runs. This is normal in MPI unless you explicitly serialize output.
 *
 * Common failure modes and troubleshooting tips
 * ---------------------------------------------
 * 1) "mpi.h: No such file or directory"
 *    - Cause: compiling without mpicc / MPI not installed.
 *    - Fix: install MPI and compile with mpicc (recommended).
 *
 * 2) Linker errors referencing MPI symbols (MPI_Init, MPI_Wtime, ...)
 *    - Cause: MPI headers found, but MPI libraries not linked.
 *    - Fix: compile with mpicc or link MPI libraries correctly.
 *
 * 3) Program appears to “hang”
 *    - In MPI, hangs often come from mismatched collectives or receives.
 *    - Here, the only collectives are MPI_Barrier and MPI_Reduce, and every rank calls them.
 *    - If you modify the code so only some ranks call these collectives, deadlock is likely.
 *
 * Correctness and safety notes
 * ----------------------------
 * - Timing:
 *   MPI_Wtime returns wall-clock time local to the process. Different ranks may have
 *   slightly different clocks; we only use differences (finish-start) locally, which is safe.
 *
 * - Synchronization:
 *   MPI_Barrier ensures all ranks reach the same point before starting timing, but does
 *   not guarantee they start at the exact same CPU cycle. It is still the correct pattern
 *   for “start timing after everyone is ready”.
 *
 * - “volatile” usage:
 *   The variable `dummy` is declared volatile to discourage aggressive compiler optimization
 *   that might remove the loop as “dead code”. This helps ensure the loop actually runs.
 *   (This is not a perfect guarantee in all circumstances, but it is a common technique.)
 *
 * - Overflow:
 *   The loop counter uses `long` and the upper bound uses a long literal (10000000L).
 *   With typical values of rank, this is safe, but extremely large rank counts could
 *   make (rank+1)*10000000L exceed the range of long on some platforms.
 *   In normal MPI runs (tens/hundreds of ranks), this is not a concern.
 */

/* ---------------------------------- Includes ---------------------------------- */

/*
 * <stdio.h>
 * - Standard C I/O library.
 * - Provides printf for printing formatted output to the console.
 */
#include <stdio.h>

/*
 * <mpi.h>
 * - MPI API header.
 * - Provides declarations for MPI_Init, MPI_Comm_rank, MPI_Barrier, MPI_Wtime, MPI_Reduce, etc.
 */
#include <mpi.h>

/* ----------------------------------- main ------------------------------------- */

int main(int argc, char *argv[])
{
    /* ============================ Phase 1: Variables =========================== */

    /*
     * int rank;
     * - This process’s rank (ID) in MPI_COMM_WORLD.
     * - Set by MPI_Comm_rank after MPI_Init.
     */
    int rank;

    /*
     * int size;
     * - Total number of processes in MPI_COMM_WORLD.
     * - Set by MPI_Comm_size after MPI_Init.
     */
    int size;

    /*
     * Timing variables (all are doubles because MPI_Wtime returns double seconds).
     *
     * local_start:
     * - Timestamp just before the workload on this rank.
     *
     * local_finish:
     * - Timestamp just after the workload on this rank.
     *
     * local_elapsed:
     * - Computed difference: local_finish - local_start.
     *
     * elapsed:
     * - The maximum elapsed time across all ranks, reduced onto rank 0.
     * - Only meaningful on rank 0 after MPI_Reduce completes.
     */
    double local_start;
    double local_finish;
    double local_elapsed;
    double elapsed;

    /* ============================ Phase 2: MPI setup =========================== */

    /*
     * MPI_Init(&argc, &argv);
     *
     * Initializes MPI. Must be called before most MPI operations.
     * MPI may inspect/modify argv for MPI-related runtime flags.
     */
    MPI_Init(&argc, &argv);

    /*
     * MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     *
     * Writes this process’s rank ID into `rank`.
     * &rank is the address where MPI stores the result.
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*
     * MPI_Comm_size(MPI_COMM_WORLD, &size);
     *
     * Writes the total number of processes into `size`.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ============================ Phase 3: Synchronize ========================= */

    /*
     * MPI_Barrier(MPI_COMM_WORLD);
     *
     * A barrier is a synchronization point:
     * - Each process that reaches the barrier waits until ALL processes reach it.
     * - Once everyone has arrived, all are released to continue.
     *
     * Why put a barrier before timing?
     * - Without it, some ranks might start timing earlier than others (e.g., due to OS scheduling),
     *   which would make comparisons less meaningful.
     *
     * Collective rule:
     * - Every rank in MPI_COMM_WORLD must call this barrier or the program may deadlock.
     */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ============================ Phase 4: Start timer ========================= */

    /*
     * local_start = MPI_Wtime();
     *
     * MPI_Wtime:
     * - Returns wall-clock time in seconds as a double.
     * - Often high-resolution (implementation-dependent).
     *
     * We store the timestamp locally; each rank times itself independently.
     */
    local_start = MPI_Wtime();

    /* ============================ Phase 5: Workload ============================ */

    /*
     * Simulated workload:
     * - We run a loop whose iteration count depends on (rank + 1).
     * - Higher ranks run longer -> deliberate load imbalance.
     *
     * Why “volatile double dummy”?
     * - If the loop computed nothing observable, an optimizing compiler might remove it.
     * - Declaring dummy volatile discourages removal by implying reads/writes have side effects.
     *
     * Caveat:
     * - “volatile” is not a full performance-model tool; it is mainly for memory-mapped I/O.
     * - But it commonly helps for keeping simple timing loops from being optimized away.
     */
    volatile double dummy = 0.0;

    /*
     * for (long i = 0; i < (rank + 1) * 10000000L; i++)
     *
     * Loop basics:
     * - i starts at 0
     * - continues while i < limit
     * - increments by 1 each iteration
     *
     * The limit:
     * - (rank + 1) * 10000000L
     * - rank=0 -> 10,000,000 iterations
     * - rank=1 -> 20,000,000 iterations
     * - etc.
     *
     * The trailing 'L' makes 10000000 a long constant, matching i’s type.
     */
    for (long i = 0; i < (rank + 1) * 10000000L; i++)
    {
        /*
         * dummy += i * 0.0000001;
         *
         * This creates some floating-point arithmetic work.
         * The exact numeric value is irrelevant; we only want measurable compute time.
         */
        dummy += i * 0.0000001;
    }

    /* ============================ Phase 6: Stop timer ========================== */

    /*
     * local_finish = MPI_Wtime();
     * - Capture timestamp after workload completes.
     */
    local_finish = MPI_Wtime();

    /*
     * local_elapsed = local_finish - local_start;
     * - Compute elapsed seconds on this rank.
     */
    local_elapsed = local_finish - local_start;

    /* ============================ Phase 7: Print local time ==================== */

    /*
     * printf("Process %d: local elapsed time = %f seconds\n", rank, local_elapsed);
     *
     * Each rank prints a line.
     *
     * Output ordering note:
     * - There is no synchronization around printf.
     * - Lines may appear in any order and can interleave (rarely mid-line, more commonly line order).
     * - This nondeterminism is due to OS scheduling and IO buffering, not MPI correctness.
     */
    printf("Process %d: local elapsed time = %f seconds\n",
           rank, local_elapsed);

    /* ============================ Phase 8: Reduce to maximum =================== */

    /*
     * MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     *
     * MPI_Reduce:
     * - A collective operation that combines one value from each rank into a single result
     *   on the root rank (here rank 0).
     *
     * Parameters:
     * - &local_elapsed: address of this rank’s contribution
     * - &elapsed: address where root stores the final reduced value
     * - 1: number of elements being reduced
     * - MPI_DOUBLE: datatype of the elements
     * - MPI_MAX: reduction operation (take the maximum across ranks)
     * - 0: root rank (receives the result)
     * - MPI_COMM_WORLD: communicator
     *
     * After this call:
     * - On rank 0: `elapsed` contains the maximum time across all ranks.
     * - On other ranks: `elapsed` is not meaningful (MPI does not guarantee it is set).
     *
     * Collective rule:
     * - Every rank must call MPI_Reduce or the program may deadlock.
     */
    MPI_Reduce(&local_elapsed,
               &elapsed,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);

    /* ============================ Phase 9: Root prints max ===================== */

    /*
     * if (rank == 0) { ... }
     *
     * Only rank 0 prints the maximum because only rank 0 receives it in MPI_Reduce.
     */
    if (rank == 0)
    {
        /*
         * Print maximum elapsed time and the number of processes.
         * This is the typical “effective parallel runtime” metric.
         */
        printf("\nMaximum elapsed time across %d processes: %f seconds\n",
               size, elapsed);
    }

    /* ============================ Phase 10: MPI cleanup ======================== */

    /*
     * MPI_Finalize();
     * - Cleanly shut down the MPI runtime.
     * - After finalize, most MPI calls are not allowed.
     */
    MPI_Finalize();

    /*
     * return 0;
     * - Indicate successful program completion.
     */
    return 0;
}
