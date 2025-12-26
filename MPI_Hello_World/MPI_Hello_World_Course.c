/*
 * =====================================================================================
 * Teaching Version (Heavily Commented): MPI Greeting Exchange (Point-to-Point Messaging)
 * =====================================================================================
 *
 * Purpose of this program
 * -----------------------
 * This program demonstrates the most basic form of MPI (Message Passing Interface) communication:
 * one process (rank 0) receives a greeting string from every other process (ranks 1..N-1).
 * Each non-zero process sends exactly one message, and rank 0 prints them.
 *
 * High-level workflow
 * ----------------------------------
 * 1) Start MPI (MPI_Init) so multiple processes can cooperate.
 * 2) Discover how many processes exist (MPI_Comm_size) and which one we are (MPI_Comm_rank).
 * 3) If we are NOT rank 0:
 *      - Build a greeting string in a local character buffer.
 *      - Send that string to rank 0 (MPI_Send).
 *    If we ARE rank 0:
 *      - Print our own greeting.
 *      - Receive one greeting from each other rank (MPI_Recv) and print it.
 * 4) Shut down MPI cleanly (MPI_Finalize).
 *
 * What this program is (and is not)
 * ---------------------------------
 * - MPI is "Message Passing Interface": a standard library for writing programs that run
 *   as multiple *separate processes* (not threads) that communicate by sending messages.
 * - Each MPI process has its *own memory*. This is crucial:
 *     - The array `gret` in rank 1 is NOT the same memory as `gret` in rank 0.
 *     - Data is shared only by explicit MPI calls (like MPI_Send / MPI_Recv).
 *
 * ASCII mental model: processes and message flow
 * ----------------------------------------------
 *
 *   Rank 1  ----send "Greets from process 1 of P!"----\
 *   Rank 2  ----send "Greets from process 2 of P!"-----+--> Rank 0 prints all greetings
 *   Rank 3  ----send "Greets from process 3 of P!"----/
 *   ...
 *
 * Build / compile instructions
 * ----------------------------
 * IMPORTANT: MPI programs must be compiled with an MPI "wrapper compiler"
 * so that the correct header files and libraries are used automatically.
 * The wrapper compiler is typically:
 *   - mpicc   (for C)
 *   - mpic++  (for C++)
 *
 * Linux (Open MPI or MPICH):
 *   mpicc -O2 -Wall MPI_Hello_World_Course.c -o MPI_Hello_World_Course
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
 *     gcc MPI_Hello_World_Course.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Hello_World_Course
 *
 * Run instructions (with examples)
 * --------------------------------
 * MPI programs are started with a launcher that creates multiple processes:
 *   mpiexec -n <num_processes> <program>
 *
 * Examples:
 *   mpiexec -n 4 MPI_Hello_World_Course
 *   mpiexec -n 8 MPI_Hello_World_Course
 *
 * Expected inputs / outputs
 * -------------------------
 * Inputs:
 *   - No user input. The program’s behavior depends only on how many processes you start.
 *
 * Outputs:
 *   - Rank 0 prints one line for itself, then one line per other process.
 *   - Total printed lines = number of processes (csize).
 *   - Example (with 4 processes, output order from rank 0 is deterministic here):
 *       Greets from process 0 of 4!
 *       Greets from process 1 of 4!
 *       Greets from process 2 of 4!
 *       Greets from process 3 of 4!
 *
 * Common failure modes and troubleshooting tips
 * ---------------------------------------------
 * 1) "mpi.h: No such file or directory"
 *    - You compiled with `gcc` instead of `mpicc`, or MPI is not installed/configured.
 *    - Fix: use `mpicc` (recommended), or add the correct include path for mpi.h.
 *
 * 2) Linker errors: "undefined reference to MPI_Init" (or similar)
 *    - You included mpi.h but did not link the MPI libraries.
 *    - Fix: compile with `mpicc`, or add the correct -L and -I flags for your MPI.
 *
 * 3) Runtime errors: mpiexec not found / cannot start processes
 *    - The MPI runtime tools are not in PATH, or the MPI installation is incomplete.
 *    - Fix: ensure `mpiexec` is installed and available in your environment.
 *
 * 4) Hanging / deadlock
 *    - MPI programs can hang if a receive waits for a message that never arrives,
 *      or tags/sources do not match.
 *    - In this program, each non-zero rank sends exactly one message to rank 0,
 *      and rank 0 receives exactly one from each sender, so it should not hang
 *      unless the program is modified incorrectly.
 *
 * Correctness and safety notes (important pitfalls)
 * -------------------------------------------------
 * - Buffer size safety:
 *   - We store messages in `char gret[MAX_STRING]`.
 *   - If the produced greeting exceeds MAX_STRING-1 characters, `sprintf` would overflow
 *     the buffer (undefined behavior: could crash or corrupt memory).
 *   - In this specific message format and typical integer sizes, MAX_STRING=100 is ample.
 *     However, using `snprintf` would be safer. We intentionally DO NOT change to snprintf
 *     because (a) we must preserve behavior, and (b) the original algorithm uses sprintf.
 *
 * - Matching send/receive:
 *   - MPI_Recv specifies source rank q and tag 0.
 *   - MPI_Send sends to destination rank 0 with tag 0.
 *   - If you change tags or sources, they must still match or rank 0 may block forever.
 *
 * - Determinism:
 *   - Rank 0 receives in a fixed order q=1..csize-1, so printed order is deterministic
 *     (assuming all sends complete eventually).
 *   - In general MPI programs, message arrival order can be nondeterministic if you use
 *     MPI_ANY_SOURCE or multiple messages; this code avoids that complexity.
 *
 * Optional enhancements (not enabled here)
 * ----------------------------------------
 * - Input validation: not applicable (there is no input).
 * - Debug prints behind a compile-time flag could be added, but we will not add extra
 *   output because it would change the program’s observable behavior.
 */

/* ------------------------------- Include files ---------------------------------- */

/*
 * <stdio.h>
 * - Standard I/O library for C.
 * - Provides functions like printf() for printing text to the console.
 * - If omitted, the compiler would not know the declaration of printf().
 */
#include <stdio.h>

/*
 * <string.h>
 * - Standard C string library.
 * - Provides strlen(), which counts characters in a C string (up to '\0').
 * - If omitted, the compiler would not know the declaration of strlen().
 */
#include <string.h>

/*
 * <mpi.h>
 * - The MPI library header file.
 * - Declares all MPI functions (MPI_Init, MPI_Comm_rank, MPI_Send, etc.)
 *   and MPI types/constants (MPI_COMM_WORLD, MPI_CHAR, MPI_STATUS_IGNORE).
 * - If omitted, none of the MPI symbols would be declared and compilation fails.
 */
#include <mpi.h>

/* ----------------------------- Global constants -------------------------------- */

/*
 * MAX_STRING
 * - A constant integer used to size the character buffer for greetings.
 * - The buffer must be big enough to hold:
 *     - the characters of the greeting message, AND
 *     - the terminating null byte '\0' used by C strings.
 *
 * Why 100?
 * - It is "comfortably large" for the small greeting we construct here.
 *
 * What if this is too small?
 * - If the greeting does not fit, `sprintf` would overflow the buffer (undefined behavior).
 * - If you reduce this number drastically, you risk memory corruption.
 */
const int MAX_STRING = 100;

/* ------------------------------- main program ---------------------------------- */

/*
 * main(void)
 * - The entry point of a C program.
 * - `void` inside parentheses means: this program does not accept command-line arguments.
 *   (Contrast with `int main(int argc, char *argv[])` which would accept them.)
 *
 * Return value:
 * - Returning 0 conventionally means "success" to the operating system / shell.
 */
int main(void)
{
+
    /*
     * char gret[MAX_STRING];
     * - Declares an array of MAX_STRING characters (bytes).
     * - Used as a buffer to store a C string (null-terminated text).
     *
     * Important concept: "C strings"
     * - In C, a string is an array of characters ending with a special byte '\0'.
     * - Functions like printf("%s", ...) and strlen(...) rely on this '\0' terminator.
     *
     * Memory note:
     * - Each MPI process has its own *private* memory.
     * - So each process has its own independent `gret` buffer.
     */
    char gret[MAX_STRING];

    /*
     * int csize;
     * - Will store the total number of MPI processes participating in MPI_COMM_WORLD.
     *
     * csize typically means "communicator size" (the number of ranks).
     * It is an ordinary C integer variable, uninitialized at this point.
     * If we tried to use it before MPI_Comm_size sets it, we would have garbage.
     */
    int csize;

    /*
     * int prank;
     * - Will store the "rank" (ID) of this process inside MPI_COMM_WORLD.
     *
     * Rank:
     * - An integer label assigned by MPI, usually from 0 to csize-1.
     * - Rank 0 is often used as the "root" coordinator by convention.
     *
     * Again: uninitialized until MPI_Comm_rank sets it.
     */
    int prank;

    /*
     * MPI_Init(NULL, NULL);
     * - Initializes the MPI runtime.
     * - Before calling MPI_Init, most MPI calls are illegal/undefined.
     *
     * Why does MPI_Init take arguments?
     * - Many MPI programs pass `&argc, &argv` so MPI can inspect/strip MPI-specific
     *   command-line arguments.
     * - This program does not accept command-line arguments, so it passes NULL.
     *
     * What happens if omitted?
     * - Subsequent MPI calls (MPI_Comm_size, MPI_Send, etc.) are invalid and may fail.
     */
    MPI_Init(NULL, NULL);

    /*
     * MPI_Comm_size(MPI_COMM_WORLD, &csize);
     *
     * MPI_COMM_WORLD:
     * - The default communicator containing all processes launched together by mpiexec.
     * - A "communicator" is an MPI concept representing a group of processes that can
     *   communicate with each other.
     *
     * &csize:
     * - "Address-of" operator in C.
     * - It produces a pointer to the variable csize, i.e., where in memory csize lives.
     * - MPI needs this address so it can *write the result into csize*.
     *
     * If you passed `csize` instead of `&csize`:
     * - MPI would interpret the integer value as an address → almost certainly crash.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    /*
     * MPI_Comm_rank(MPI_COMM_WORLD, &prank);
     * - Similar to MPI_Comm_size, but returns this process's rank ID.
     *
     * Rank range:
     * - prank is guaranteed (by MPI) to be in [0, csize - 1].
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /* ===================== Phase 2: Decide role by rank ====================== */

    /*
     * if (prank != 0) { ... } else { ... }
     *
     * Control structure: if/else
     * - Executes one block if the condition is true, otherwise executes the else block.
     *
     * Here, the program chooses roles:
     * - prank != 0: "sender" processes
     * - prank == 0: "collector/root" process
     *
     * This is a common MPI pattern: one root gathers or coordinates work.
     */
    if (prank != 0) {

        /* =================== Phase 3A: Sender (rank != 0) ==================== */

        /*
         * sprintf(gret, "Greets from process %d of %d!", prank, csize);
         *
         * sprintf:
         * - Writes formatted text into a character array (buffer) instead of printing it.
         * - Similar format rules as printf:
         *     %d means "insert an integer here".
         *
         * gret:
         * - The destination buffer where the characters will be written.
         *
         * Why create a string first?
         * - Because we want to send a message (bytes) to another process.
         *
         * Important safety note:
         * - sprintf does NOT check buffer length.
         * - If the formatted text is longer than MAX_STRING-1, overflow occurs.
         * - The original program accepts that risk; we preserve it to keep behavior identical.
         */
        sprintf(gret, "Greets from process %d of %d!", prank, csize);

        /*
         * MPI_Send(gret, strlen(gret) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
         *
         * MPI_Send:
         * - A blocking send operation in MPI point-to-point communication.
         * - "Blocking" here means: the call does not return until it is safe for the
         *   program to reuse/modify the send buffer (details depend on MPI implementation).
         *
         * Parameters explained:
         *
         * 1) gret
         *    - In C, an array name usually "decays" to a pointer to its first element.
         *    - So `gret` here effectively means: address of gret[0].
         *
         * 2) strlen(gret) + 1
         *    - strlen counts characters up to (but not including) '\0'.
         *    - We add 1 to include the '\0' terminator in the message.
         *
         *    Why send the '\0'?
         *    - Rank 0 will treat received bytes as a C string and print it with "%s".
         *    - If we did NOT send '\0', the receiver might print garbage or crash,
         *      because it would not know where the string ends.
         *
         * 3) MPI_CHAR
         *    - Tells MPI that each element in the buffer is a char.
         *
         * 4) 0
         *    - Destination rank: we send to rank 0 (the collector).
         *
         * 5) 0
         *    - Tag: an integer used to label messages.
         *    - The receiver can use tags to select which message to receive.
         *    - This must match the tag used in MPI_Recv (below), otherwise the receive
         *      will not match and may block.
         *
         * 6) MPI_COMM_WORLD
         *    - Communicator across which we communicate.
         *
         * What if we changed destination or tag?
         * - Rank 0 expects exactly one message from each rank q with tag 0.
         * - Any mismatch can cause rank 0 to wait forever (deadlock).
         */
        MPI_Send(gret, strlen(gret) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

    } else {

        /* =================== Phase 3B: Collector (rank == 0) ================= */

        /*
         * printf("Greets from process %d of %d!\n", prank, csize);
         *
         * Rank 0 prints its own greeting without communicating.
         *
         * Note:
         * - Only rank 0 prints; other ranks do not print anything.
         * - This prevents interleaved output from many processes, which can be messy.
         */
        printf("Greets from process %d of %d!\n", prank, csize);

        /*
         * for (int q = 1; q < csize; q++) { ... }
         *
         * Control structure: for-loop
         * - Initializes q=1.
         * - Repeats as long as q < csize.
         * - Increments q by 1 after each iteration.
         *
         * Why start at 1?
         * - Rank 0 already printed its own greeting.
         * - The remaining ranks are 1..csize-1, each sending exactly one message.
         *
         * Edge case:
         * - If csize == 1 (you launched only 1 process), the loop runs zero times.
         * - Output will only be the rank 0 greeting. This is valid and does not hang.
         */
        for (int q = 1; q < csize; q++) {

            /*
             * MPI_Recv(gret, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             *
             * MPI_Recv:
             * - A blocking receive operation.
             * - It waits until a matching message arrives, then writes received bytes
             *   into the provided buffer.
             *
             * Parameters explained:
             *
             * 1) gret
             *    - Receive buffer (where MPI will store incoming bytes).
             *    - Again: `gret` means the address of the first element.
             *
             * 2) MAX_STRING
             *    - Maximum number of MPI_CHAR elements MPI is allowed to write into gret.
             *    - This protects us from writing beyond the buffer size.
             *
             * Important note about truncation:
             * - If a sender transmits more than MAX_STRING chars, behavior depends on MPI:
             *   it may raise an error or truncate, depending on settings.
             * - In this program, senders transmit strlen+1 of a short message, so it fits.
             *
             * 3) MPI_CHAR
             *    - The datatype of each element we receive.
             *
             * 4) q
             *    - Source rank. We are explicitly expecting the message from rank q.
             *    - This makes receive order deterministic (q=1, then q=2, ...).
             *
             * 5) 0
             *    - Tag. Must match sender’s tag in MPI_Send.
             *
             * 6) MPI_COMM_WORLD
             *    - Communicator used.
             *
             * 7) MPI_STATUS_IGNORE
             *    - MPI can optionally return a "status" describing what was received
             *      (actual source, tag, and message length).
             *    - We do not need that information, so we tell MPI to ignore it.
             *
             * What if omitted?
             * - Rank 0 would not receive the message and the sender might still complete,
             *   but rank 0 would not print anything and/or could deadlock if logic changes.
             */
            MPI_Recv(gret, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /*
             * printf("%s\n", gret);
             *
             * %s expects a null-terminated C string.
             *
             * Why is gret a valid string here?
             * - The sender included '\0' by sending strlen(gret)+1.
             *
             * What if the sender did NOT send '\0'?
             * - printf would continue reading memory past the buffer until it
             *   accidentally finds a '\0' somewhere ("works by accident"),
             *   or it could crash ("undefined behavior").
             *
             * The program’s correctness relies on:
             * - The sender sending the terminator, and
             * - The receiver having a big enough buffer.
             */
            printf("%s\n", gret);
        }
    }

    /* ============================ Phase 4: Cleanup =========================== */

    /*
     * MPI_Finalize();
     * - Shuts down the MPI runtime environment for this process.
     *
     * After MPI_Finalize:
     * - No MPI calls are permitted (they are invalid).
     *
     * Why finalize?
     * - Ensures MPI can clean up internal resources (network connections, shared memory,
     *   process management bookkeeping, etc.).
     * - Some MPI implementations may behave badly if you omit it.
     */
    MPI_Finalize();

    /*
     * return 0;
     * - Reports successful completion to the operating system.
     */
    return 0;
}
