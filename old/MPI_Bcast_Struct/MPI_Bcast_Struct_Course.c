/*
 * =====================================================================================
 * MPI_Bcast_Struct_Course.c  —  Teaching Version (Heavily Commented, Beginner-First)
 * =====================================================================================
 *
 * Purpose (1–3 sentences)
 * -----------------------
 * This program demonstrates how to broadcast (send from one process to all processes)
 * a C `struct` using MPI. Because C structs may contain compiler-inserted padding bytes,
 * we create an MPI “derived datatype” that exactly matches the in-memory layout of the
 * struct and then use MPI_Bcast to distribute one struct instance from rank 0 to all ranks.
 *
 * High-level workflow (bullet steps)
 * ----------------------------------
 * 1) Initialize MPI (MPI_Init), discover rank count (MPI_Comm_size) and our rank (MPI_Comm_rank).
 * 2) Define a C struct (SData) and build an MPI derived datatype (MPI_Type_create_struct)
 *    that describes the struct’s real memory layout (including padding/alignment).
 * 3) Rank 0 reads struct field values from stdin (scanf).
 * 4) Broadcast the struct from rank 0 to all ranks (MPI_Bcast).
 * 5) Every rank prints the struct it now holds.
 * 6) Clean up: free the MPI datatype (MPI_Type_free) and finalize MPI (MPI_Finalize).
 *
 * What MPI_Bcast means (conceptual)
 * ---------------------------------
 * - MPI programs run as multiple *processes* (separate OS processes), usually launched via mpiexec.
 * - Each process has its own memory; a variable in rank 1 is NOT the same memory as a variable
 *   in rank 0. Data is shared only through MPI calls.
 *
 * MPI_Bcast is a “collective operation”:
 * - “Collective” means: ALL ranks in the communicator must call it.
 * - One designated rank (the “root”) provides the initial data.
 * - After the call returns, every rank has an identical copy of that data in its own memory.
 *
 * ASCII picture of the broadcast (root = rank 0)
 * ----------------------------------------------
 *
 *        (rank 0)             (rank 1)   (rank 2)   (rank 3)
 *       +---------+           +------+    +------+    +------+
 *       |  SData  |  ----->   |SData |    |SData |    |SData |
 *       +---------+           +------+    +------+    +------+
 *            \
 *             \---> (conceptually: root distributes the same bytes to everyone)
 *
 * Why a derived datatype is necessary for structs
 * ----------------------------------------------
 * In C, a struct’s in-memory layout is not always “packed” field-by-field.
 * Compilers may insert invisible padding bytes between fields to satisfy alignment rules.
 *
 * Example (not guaranteed, but common):
 *   struct { int i1; double d1; double d2; }
 * may be laid out as:
 *
 *   offset: 0          4          8          16         24
 *           +----------+----------+----------+----------+
 *           |   i1     | padding  |   d1     |   d2     |
 *           +----------+----------+----------+----------+
 *              4B          4B        8B         8B
 *
 * If you incorrectly assume “no padding”, you might describe the struct wrong to MPI,
 * causing fields to be interpreted at the wrong addresses on receive.
 *
 * Using:
 *   offsetof(SData, field)
 * ensures offsets match the compiler’s real layout on that platform/compiler/ABI.
 *
 * Build / compile instructions
 * ----------------------------
 * Use an MPI wrapper compiler if possible (recommended), because it automatically adds
 * the correct include paths and libraries:
 *
 * Linux / macOS (Open MPI or MPICH):
 *   mpicc -O2 -Wall MPI_Bcast_Struct_Course.c -o MPI_Bcast_Struct_Course
 *
 * Windows (varies by MPI installation):
 *   - If you have mpicc available, use it similarly.
 *   - If using Microsoft MPI without mpicc, you must provide include/library paths manually.
 *
 * Run instructions (with examples)
 * --------------------------------
 * Launch with an MPI runtime launcher:
 *   mpiexec -n <num_processes> <program>
 *
 * Example:
 *   mpiexec -n 4 ./MPI_Bcast_Struct_Course
 *
 * Then enter input ONLY ONCE (rank 0 reads it). Example input:
 *   42 3.14 2.718
 *
 * Expected inputs / outputs
 * -------------------------
 * Input format (read by rank 0):
 *   <int> <double> <double>
 *
 * Output (printed by every rank, order may vary):
 *   Process <rank> - Data <i1> <d1> <d2>
 *
 * Example (with 4 ranks; ordering is not guaranteed):
 *   Process 0 - Data 42 3.140000 2.718000
 *   Process 1 - Data 42 3.140000 2.718000
 *   Process 2 - Data 42 3.140000 2.718000
 *   Process 3 - Data 42 3.140000 2.718000
 *
 * Note about print ordering (determinism vs nondeterminism)
 * --------------------------------------------------------
 * Each process prints independently. The console output can interleave or appear in
 * varying order across runs because processes race to write to stdout.
 * MPI does not guarantee a global print order unless you add explicit synchronization.
 *
 * Common failure modes and troubleshooting tips
 * ---------------------------------------------
 * 1) "mpi.h: No such file or directory"
 *    - You compiled with gcc/clang instead of mpicc, or MPI is not installed.
 *    - Fix: install MPI and compile with mpicc (recommended).
 *
 * 2) Linker errors about MPI symbols (MPI_Init, MPI_Bcast, ...)
 *    - MPI headers found, but MPI libraries not linked.
 *    - Fix: use mpicc or add correct MPI libs manually.
 *
 * 3) Program hangs waiting for input
 *    - Rank 0 calls scanf and waits for input.
 *    - If you run under an environment where stdin is not connected as expected,
 *      rank 0 may block. Provide input in the launching terminal or redirect input.
 *
 * 4) Wrong results when broadcasting structs in other programs
 *    - Usually caused by using MPI_BYTE / sizeof(struct) assumptions across differing
 *      compilers/ABIs or forgetting padding/alignment. This program shows the correct
 *      approach: MPI_Type_create_struct + offsetof.
 *
 * Correctness and safety notes
 * ----------------------------
 * - This program assumes all ranks run the same executable built for the same ABI
 *   (typical in MPI). If ranks ran with different struct layouts (rare in practice),
 *   then offsets/types must still match logically.
 *
 * - scanf return value is not checked:
 *   - If the user provides invalid input, fields may remain uninitialized.
 *   - We do not add validation because it could change behavior in edge cases.
 *
 * - MPI derived datatypes must be:
 *   - created (MPI_Type_create_struct),
 *   - committed (MPI_Type_commit),
 *   - and eventually freed (MPI_Type_free).
 *   Forgetting commit can cause MPI communication calls to fail or behave unexpectedly.
 */

/* ---------------------------------- Includes ---------------------------------- */

/*
 * <stdio.h>
 * - C standard I/O.
 * - Provides printf (output) and scanf (input).
 */
#include <stdio.h>

/*
 * <mpi.h>
 * - MPI API header.
 * - Declares MPI_Init, MPI_Comm_rank, MPI_Bcast, MPI_Type_create_struct, etc.
 */
#include <mpi.h>

/*
 * <stddef.h>
 * - Provides offsetof(type, member).
 * - offsetof computes the byte offset of a struct member from the start of the struct.
 * - This is the standard, portable way to ask “where is this field located in memory?”
 */
#include <stddef.h>

/* ------------------------------ Data structure --------------------------------- */

/*
 * typedef struct SData { ... } SData;
 *
 * - Defines a “record-like” aggregate type with three fields.
 * - `typedef` creates an alias so we can write `SData` instead of `struct SData`.
 *
 * Field types:
 * - int: typically a 32-bit signed integer (exact size depends on platform, but common).
 * - double: typically an IEEE-754 64-bit floating point number (very common).
 *
 * IMPORTANT: Padding/alignment
 * - The compiler may insert padding bytes between fields.
 * - Therefore, you must not guess field offsets manually.
 * - We will compute offsets with offsetof and build a matching MPI datatype.
 */
typedef struct SData
{
    int    i1;  /* integer field */
    double d1;  /* first double field */
    double d2;  /* second double field */
} SData;

/* ----------------------------------- main ------------------------------------- */

int main(int argc, char *argv[])
{
    /* ============================ Phase 1: MPI setup ========================== */

    /*
     * int csize;
     * - Will hold the number of MPI processes in MPI_COMM_WORLD.
     * - Uninitialized until MPI_Comm_size writes into it.
     */
    int csize;

    /*
     * int prank;
     * - Will hold this process’s rank (ID) in MPI_COMM_WORLD, in [0..csize-1].
     * - Uninitialized until MPI_Comm_rank writes into it.
     */
    int prank;

    /*
     * MPI_Init(&argc, &argv);
     *
     * Initializes MPI. Must be called before most MPI functions.
     *
     * Why pass &argc, &argv?
     * - MPI may parse command-line arguments for its own options.
     * - Passing pointers lets MPI modify argc/argv if needed.
     */
    MPI_Init(&argc, &argv);

    /*
     * MPI_Comm_size(MPI_COMM_WORLD, &csize);
     * - Writes the number of processes in the communicator into csize.
     *
     * &csize:
     * - Address-of operator. MPI needs a pointer so it can store the result.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    /*
     * MPI_Comm_rank(MPI_COMM_WORLD, &prank);
     * - Writes the calling process’s rank into prank.
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /* =================== Phase 2: Build MPI datatype for SData ================= */

    /*
     * MPI_Datatype data_t;
     *
     * MPI_Datatype:
     * - An MPI “handle” representing a datatype (built-in or derived).
     * - Built-in examples: MPI_INT, MPI_DOUBLE
     * - Derived types allow MPI to send/receive complex layouts like structs.
     *
     * data_t will describe exactly how SData is laid out in memory.
     */
    MPI_Datatype data_t;

    /*
     * lengths[3] = {1,1,1};
     *
     * We describe the struct as 3 blocks (one per field).
     * Each block length is “how many items” of the corresponding MPI type.
     *
     * Here each field is a single value:
     * - i1: 1 int
     * - d1: 1 double
     * - d2: 1 double
     */
    int lengths[3] = { 1, 1, 1 };

    /*
     * MPI_Aint offsets[3];
     *
     * MPI_Aint:
     * - An integer type used by MPI to store addresses or byte displacements.
     * - It is large enough to hold pointer differences on the platform.
     *
     * offsets[k] will store the byte offset of the k-th block within SData.
     */
    MPI_Aint offsets[3];

    /*
     * MPI_Datatype types[3] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE };
     *
     * types[k] is the MPI type corresponding to the k-th block:
     * - Field i1 is an int          -> MPI_INT
     * - Field d1 is a double        -> MPI_DOUBLE
     * - Field d2 is a double        -> MPI_DOUBLE
     */
    MPI_Datatype types[3] = {
        MPI_INT,
        MPI_DOUBLE,
        MPI_DOUBLE
    };

    /*
     * Compute offsets using offsetof.
     *
     * offsetof(SData, i1) returns the number of bytes from the start of an SData object
     * to the member i1.
     *
     * Casting to MPI_Aint:
     * - offsetof returns a size_t (an unsigned integer type).
     * - MPI expects offsets as MPI_Aint.
     * - The cast is a standard way to convert the value into MPI’s displacement type.
     *
     * Why not hardcode offsets like {0, 4, 12}?
     * - Because padding/alignment rules differ by compiler/ABI.
     * - Hardcoding would be “works by accident” on one platform and wrong on another.
     */
    offsets[0] = (MPI_Aint)offsetof(SData, i1);
    offsets[1] = (MPI_Aint)offsetof(SData, d1);
    offsets[2] = (MPI_Aint)offsetof(SData, d2);

    /*
     * MPI_Type_create_struct(3, lengths, offsets, types, &data_t);
     *
     * Creates a derived datatype describing a struct layout.
     *
     * Parameters:
     * - count = 3 blocks
     * - lengths = array of block lengths
     * - offsets = array of byte displacements (from start of struct)
     * - types = array of MPI datatypes for each block
     * - &data_t = output handle where MPI stores the new datatype
     *
     * This tells MPI: “An SData consists of:
     *   - 1 MPI_INT at offsets[0],
     *   - 1 MPI_DOUBLE at offsets[1],
     *   - 1 MPI_DOUBLE at offsets[2].”
     */
    MPI_Type_create_struct(3, lengths, offsets, types, &data_t);

    /*
     * MPI_Type_commit(&data_t);
     *
     * Commits the datatype so it can be used in communication calls (MPI_Bcast, etc.).
     *
     * Conceptually:
     * - “Create” defines the layout.
     * - “Commit” finalizes it and allows MPI to build internal metadata/optimizations.
     *
     * If omitted:
     * - Using data_t in MPI communication is an error and may fail at runtime.
     */
    MPI_Type_commit(&data_t);

    /* ============================ Phase 3: Prepare data ========================= */

    /*
     * SData s;
     *
     * This is the actual struct instance we will broadcast.
     * Each rank has its own local variable `s` in its own address space.
     *
     * Before broadcast:
     * - Rank 0 will initialize s by reading input.
     * - Other ranks have an uninitialized s (garbage) until MPI_Bcast overwrites it.
     */
    SData s;

    /*
     * if (prank == 0) { ... }
     *
     * Only the root rank reads from stdin.
     * If every rank tried to read, they would compete for the same input stream,
     * leading to confusion and likely incorrect values.
     */
    if (prank == 0)
    {
        /*
         * scanf("%d %lf %lf", &s.i1, &s.d1, &s.d2);
         *
         * Reads three values in order:
         * - %d  -> int
         * - %lf -> double
         * - %lf -> double
         *
         * Addresses:
         * - &s.i1 points to the int field.
         * - &s.d1 points to the first double field.
         * - &s.d2 points to the second double field.
         *
         * If you omitted the &:
         * - scanf would attempt to write to an invalid memory address → undefined behavior.
         *
         * Input validation note:
         * - scanf returns how many items were successfully read.
         * - This code does not check it (preserving original behavior).
         * - Invalid input can leave fields uninitialized.
         */
        scanf("%d %lf %lf", &s.i1, &s.d1, &s.d2);
    }

    /* ============================ Phase 4: Broadcast ============================ */

    /*
     * MPI_Bcast(&s, 1, data_t, 0, MPI_COMM_WORLD);
     *
     * Broadcasts ONE object of type `data_t` from root rank 0 to all ranks.
     *
     * Parameters:
     * - &s: address of the buffer (the struct instance)
     * - 1: count (one struct object)
     * - data_t: the derived datatype describing SData layout
     * - 0: root rank (source of the broadcast)
     * - MPI_COMM_WORLD: communicator
     *
     * After this call returns (guaranteed by MPI semantics):
     * - Rank 0 still has its original s.
     * - Every other rank has received identical field values into its own local s.
     *
     * Collective call rule:
     * - Every rank in MPI_COMM_WORLD must call MPI_Bcast here.
     * - If some ranks skip it, the program can deadlock.
     */
    MPI_Bcast(&s, 1, data_t, 0, MPI_COMM_WORLD);

    /* ============================ Phase 5: Output =============================== */

    /*
     * printf("Process %d - Data %d %lf %lf\n", prank, s.i1, s.d1, s.d2);
     *
     * Each rank prints its rank ID and the struct fields.
     *
     * Output order:
     * - Not guaranteed. Different ranks may print in different orders across runs.
     * - If you need ordered output, you would add synchronization (e.g., a loop with barriers).
     */
    printf("Process %d - Data %d %lf %lf\n", prank, s.i1, s.d1, s.d2);

    /* ============================ Phase 6: Cleanup ============================== */

    /*
     * MPI_Type_free(&data_t);
     *
     * Releases resources associated with the derived datatype.
     *
     * Why do this?
     * - In small demos, forgetting it may not matter much.
     * - In real, long-running or library code, not freeing datatypes can leak MPI resources.
     */
    MPI_Type_free(&data_t);

    /*
     * MPI_Finalize();
     *
     * Shuts down MPI. After this call, most MPI functions are not allowed.
     * Always finalize before exiting for clean shutdown.
     */
    MPI_Finalize();

    /*
     * return 0;
     * - Conventional success exit code.
     */
    return 0;
}
