/*
 * =====================================================================================
 * MPI_Matrix_Vector_Course.c  —  Teaching Version (Heavily Commented, Beginner-First)
 * =====================================================================================
 *
 * Purpose (1–3 sentences)
 * -----------------------
 * This program performs a matrix–vector multiplication in parallel using MPI:
 *
 *     y = A * x
 *
 * where x is a vector of length dim, A is a dim×dim matrix, and y is a vector of length dim.
 * The matrix rows are distributed across MPI processes; each process computes a subset of
 * output entries, and rank 0 gathers the final result and writes it to a file.
 *
 * High-level workflow (bullet steps)
 * ----------------------------------
 * 1) Initialize MPI; discover how many processes are running (csize) and our rank (prank).
 * 2) Rank 0 determines dim by counting how many doubles are in the vector file.
 * 3) Broadcast dim to all ranks (MPI_Bcast).
 * 4) Rank 0 loads the full vector; broadcast the vector to all ranks (MPI_Bcast).
 * 5) Rank 0 loads the full matrix; scatter contiguous row blocks to all ranks (MPI_Scatter).
 * 6) Each rank computes its partial output y for its assigned rows.
 * 7) Gather partial outputs on rank 0 (MPI_Gather).
 * 8) Rank 0 writes the final result to "Result.txt".
 * 9) Free memory and finalize MPI.
 *
 * MPI concepts used
 * -----------------
 * - Processes (not threads): MPI runs multiple OS processes; each has its own memory.
 * - Rank: each process has an integer ID in [0..csize-1].
 * - Collectives:
 *     - MPI_Bcast  : one-to-all broadcast (all ranks must call).
 *     - MPI_Scatter: root distributes chunks to all ranks (all ranks must call).
 *     - MPI_Gather : all ranks send chunks back to root (all ranks must call).
 *
 * Data layout assumptions (very important)
 * ----------------------------------------
 * 1) Vector file:
 *    - Plain text file containing one double per entry (whitespace-separated).
 *    - Example content for dim=4:
 *         1.0 2.0 3.0 4.0
 *
 * 2) Matrix file:
 *    - Plain text file containing dim*dim doubles.
 *    - Stored in row-major order (C style):
 *         A[0,0] A[0,1] ... A[0,dim-1]
 *         A[1,0] A[1,1] ... A[1,dim-1]
 *         ...
 *
 * 3) Divisibility requirement:
 *    - This program assumes dim is divisible by csize.
 *    - If not divisible, the scatter/gather partitioning is wrong and results are invalid.
 *    - We DO NOT add checks that abort, because changing failure behavior could change
 *      “behavior” in edge cases. Instead, we document the requirement explicitly.
 *
 * Build / compile instructions
 * ----------------------------
 * NOTE: This file uses C++-style dynamic allocation (`new[]` / `delete[]`), even though
 * the file extension suggests C. Therefore, compile with an MPI C++ wrapper compiler.
 *
 * Linux / macOS (Open MPI or MPICH):
 *   mpicxx -O2 -Wall MPI_Matrix_Vector_Course.c -o MPI_Matrix_Vector_Course
 *   (Some systems require .cpp for C++ compilation; if so, rename accordingly.)
 *
 * Windows:
 *   - Use your MPI C++ wrapper if available (mpicxx).
 *   - Otherwise, compile with a C++ compiler and link MPI libraries manually
 *     (exact flags depend on your MPI distribution).
 *
 * Run instructions (with examples)
 * --------------------------------
 * Launch with mpiexec (creates multiple processes):
 *   mpiexec -n <num_processes> <program> <vector_file> <matrix_file>
 *
 * Example:
 *   mpiexec -n 4 ./MPI_Matrix_Vector_Course vec.txt mat.txt
 *
 * Expected inputs / outputs
 * -------------------------
 * Inputs:
 *   - argv[1]: vector file path
 *   - argv[2]: matrix file path
 *
 * Output:
 *   - A file "Result.txt" written by rank 0 containing dim doubles (one line),
 *     each followed by a space.
 *
 * Common failure modes and troubleshooting tips
 * ---------------------------------------------
 * 1) Crash due to missing command-line arguments
 *    - This program uses argv[1] and argv[2] without checking argc.
 *    - If you run it without two file paths, it will dereference invalid pointers.
 *    - Fix: always provide two arguments: vector file then matrix file.
 *
 * 2) File open fails (fopen returns NULL)
 *    - This program does not check fopen return values.
 *    - If a file path is wrong or permissions are missing, fscanf will likely crash.
 *    - Fix: ensure files exist and are readable from the working directory.
 *
 * 3) dim not divisible by csize
 *    - Results will be wrong because the program assumes equal row blocks per rank.
 *    - Fix: choose number of ranks that divides dim (e.g., dim=100 and csize=4).
 *
 * 4) Wrong file content (non-numeric tokens, too few numbers)
 *    - fscanf will fail early; arrays may be only partially filled; results meaningless.
 *    - Fix: ensure each file contains the required number of doubles.
 *
 * Determinism vs nondeterminism
 * -----------------------------
 * - Numerical results are deterministic given identical inputs and rank count because
 *   computation is fixed and reduction is gather-based (not associative reductions here).
 * - Output ordering to stdout is minimal (only rank 0 writes Result.txt). No interleaving.
 *
 * Correctness and safety notes (pitfalls)
 * ---------------------------------------
 * - Memory management:
 *   - Uses new[] / delete[] (C++). Mismatching with malloc/free would be incorrect.
 * - No input validation:
 *   - Missing argv or unreadable files can cause undefined behavior or crashes.
 * - Floating-point:
 *   - Matrix-vector multiplication accumulates rounding error in each dot product.
 * - Large dim:
 *   - Rank 0 loads the entire matrix into memory; may not fit for huge matrices.
 * - Return types and naming:
 *   - Functions accept `char*` file names. Passing string literals is OK in C++ but
 *     would ideally be `const char*`. We keep original signatures to preserve behavior.
 */

/* ---------------------------------- Includes ---------------------------------- */

/*
 * <stdio.h>
 * - Provides C file I/O functions:
 *     FILE*, fopen, fscanf, fprintf, fclose
 * - Also provides standard types and IO functionality.
 */
#include <stdio.h>

/*
 * <mpi.h>
 * - MPI API header.
 * - Declares MPI_Init, MPI_Comm_rank, MPI_Bcast, MPI_Scatter, MPI_Gather, etc.
 */
#include <mpi.h>

/* =============================== Helper: returnSize ============================= */

/*
 * returnSize
 * ----------
 * Reads a text file containing doubles and counts how many doubles it contains.
 * That count is treated as the vector dimension (dim).
 *
 * Parameters:
 *   fname - path to a text file containing double values separated by whitespace.
 *
 * Returns:
 *   The number of successfully scanned doubles (dim).
 *
 * IMPORTANT pitfalls:
 * - The code does not check whether fopen succeeded.
 *   If fopen returns NULL (file missing), fscanf will dereference NULL -> crash.
 * - The loop uses `fscanf(...) != EOF`. For malformed input (non-numeric tokens),
 *   fscanf returns 0 (not EOF), which can cause an infinite loop because the file
 *   position does not advance. This is a classic “works for valid files” pattern.
 * - We do not change it to preserve behavior; we only document it.
 */
int returnSize(char* fname)
{
    /*
     * FILE* f = fopen(fname, "r");
     *
     * FILE*:
     * - An opaque handle representing an open file stream.
     *
     * fopen(..., "r"):
     * - Opens file for reading.
     * - Returns NULL on failure (not checked here).
     */
    FILE* f = fopen(fname, "r");

    /*
     * int dim = 0;
     * - Counter for how many doubles we read.
     * - Starts at 0 and increments once per successful scan.
     */
    int dim = 0;

    /*
     * double tmp;
     * - Temporary variable to store each scanned value.
     * - We do not keep these values; we only count them.
     */
    double tmp;

    /*
     * while (fscanf(f, "%lf", &tmp) != EOF) dim++;
     *
     * fscanf returns:
     * - EOF (typically -1) when end-of-file occurs before any conversion.
     * - number of items successfully converted (1 for a valid double).
     *
     * For well-formed files containing only doubles, this counts how many exist.
     */
    while (fscanf(f, "%lf", &tmp) != EOF)
        dim++;

    /*
     * fclose(f);
     * - Releases the file handle and flushes buffers (for output files).
     * - Here it simply closes the read stream.
     */
    fclose(f);

    /*
     * return dim;
     * - The vector length.
     */
    return dim;
}

/* ================================ Helper: loadVec =============================== */

/*
 * loadVec
 * -------
 * Allocates a dynamic array of n doubles and fills it with values from a text file.
 *
 * Parameters:
 *   fname - path to file containing at least n double values
 *   n     - number of doubles to allocate and (ideally) load
 *
 * Returns:
 *   Pointer to newly allocated array of n doubles.
 *
 * Memory ownership:
 * - Caller must free the returned array using delete[].
 *
 * Pitfalls:
 * - No check for fopen success.
 * - Reads until EOF, not “exactly n items”.
 *   If the file has fewer values, part of the array remains uninitialized.
 *   If the file has more values, it will write past allocated memory -> undefined behavior.
 * - We keep this behavior to preserve the original code’s semantics.
 */
double* loadVec(char* fname, int n)
{
    /* Open file for reading (not checked for NULL). */
    FILE* f = fopen(fname, "r");

    /*
     * double* res = new double[n];
     *
     * new[] (C++):
     * - Allocates an array of n doubles on the heap (dynamic memory).
     * - Returns a pointer to the first element.
     *
     * Why heap allocation?
     * - n may be large; stack allocation could overflow.
     */
    double* res = new double[n];

    /*
     * double* it = res;
     *
     * Pointer `it` will walk through the allocated array.
     * Initially it points at res[0].
     */
    double* it = res;

    /*
     * while (fscanf(f, "%lf", it++) != EOF) ;
     *
     * This idiom:
     * - reads a double into *it
     * - then increments the pointer to the next element
     *
     * “empty body” loop:
     * - The work is done inside the condition expression.
     *
     * Pitfall:
     * - No bound check against n. If file contains > n doubles -> overflow.
     */
    while (fscanf(f, "%lf", it++) != EOF)
        /* empty body */;

    /* Close file. */
    fclose(f);

    /* Return the allocated and filled array. */
    return res;
}

/* ================================ Helper: loadMat =============================== */

/*
 * loadMat
 * -------
 * Allocates and loads an n×n matrix from a text file.
 *
 * Storage representation:
 * - The matrix is stored as a 1D array in row-major order:
 *     res[i * n + j] corresponds to A[i][j].
 *
 * Parameters:
 *   fname - path to the matrix file
 *   n     - dimension (matrix is n×n)
 *
 * Returns:
 *   Pointer to newly allocated array of n*n doubles (caller must delete[]).
 *
 * Same pitfalls as loadVec:
 * - no fopen NULL check
 * - reads until EOF with no bound checks
 * - assumes file content is valid
 */
double* loadMat(char* fname, int n)
{
    /* Open file for reading. */
    FILE* f = fopen(fname, "r");

    /*
     * Allocate n*n doubles.
     * For dim=1000, this is 1,000,000 doubles (~8 MB).
     * For larger dims, memory grows as O(n^2).
     */
    double* res = new double[n * n];

    /* Pointer iterator to fill res sequentially. */
    double* it = res;

    /* Read values sequentially into the array until EOF. */
    while (fscanf(f, "%lf", it++) != EOF)
        /* empty body */;

    /* Close file. */
    fclose(f);

    /* Return pointer to allocated matrix storage. */
    return res;
}

/* ================================= Helper: logRes =============================== */

/*
 * logRes
 * ------
 * Writes the result vector to a text file.
 *
 * Parameters:
 *   fname - output file path
 *   res   - pointer to vector of length n
 *   n     - number of elements to write
 *
 * Output format:
 * - One line containing n values, each followed by a space.
 *
 * Pitfalls:
 * - fopen is not checked for NULL.
 * - fprintf formatting uses %lf (double).
 */
void logRes(const char* fname, double* res, int n)
{
    /* Open file for writing ("w" truncates existing file). */
    FILE* f = fopen(fname, "w");

    /* Write each element followed by a space. */
    for (int i = 0; i != n; ++i)
        fprintf(f, "%lf ", res[i]);

    /* Close file. */
    fclose(f);
}

/* ===================================== main ==================================== */

int main(int argc, char* argv[])
{
    /* ============================ Phase 1: MPI setup =========================== */

    /*
     * int csize, prank;
     * - csize: total number of ranks in MPI_COMM_WORLD.
     * - prank: this rank’s ID (0..csize-1).
     */
    int csize;
    int prank;

    /*
     * MPI_Init(&argc, &argv);
     * - Initialize MPI runtime.
     * - Must be called before most MPI calls.
     */
    MPI_Init(&argc, &argv);

    /* Ask MPI how many processes exist in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &csize);

    /* Ask MPI what rank ID this process has. */
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /* ====================== Phase 2: Read command-line args ===================== */

    /*
     * Command line arguments expected:
     *   argv[1] = vector file path
     *   argv[2] = matrix file path
     *
     * IMPORTANT:
     * - This program does NOT check argc.
     * - If argc < 3, argv[1] or argv[2] is invalid -> undefined behavior / crash.
     * - We keep this unchanged to preserve behavior for valid inputs.
     */
    char* vfname = argv[1];
    char* mfname = argv[2];

    /* ======================= Phase 3: Declare key variables ===================== */

    /*
     * int dim;
     * - Global dimension of vector and matrix.
     * - Determined by counting values in the vector file (rank 0), then broadcast.
     */
    int dim;

    /*
     * Pointers for dynamic arrays:
     *
     * mat  : local chunk of the matrix (each rank receives a different part)
     * vec  : full vector (broadcast to all ranks, so everyone has same vec)
     * tmat : full matrix (only rank 0 owns it, used as scatter source)
     * lres : local result vector (partial y values computed by this rank)
     * res  : full result vector (only rank 0 allocates it, gather target)
     *
     * Why pointers?
     * - Because we allocate arrays whose size depends on dim at runtime.
     */
    double* mat;
    double* vec;
    double* tmat;
    double* lres;
    double* res;

    /* =================== Phase 4: Determine and broadcast dimension ============= */

    /*
     * Only rank 0 reads the vector file to determine dim.
     * This avoids all ranks reading the same file just to count values.
     */
    if (prank == 0)
        dim = returnSize(vfname);

    /*
     * Broadcast dim to all ranks.
     * After this:
     * - Every rank has the same dim value and can allocate correct buffer sizes.
     */
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ========================= Phase 5: Load/broadcast vector =================== */

    /*
     * Load vector on rank 0, allocate on others.
     *
     * Root (rank 0) will read from file.
     * Others just allocate memory and wait to receive vector data via MPI_Bcast.
     */
    if (prank == 0)
        vec = loadVec(vfname, dim);
    else
        vec = new double[dim];

    /*
     * Broadcast the vector data from rank 0 to all ranks.
     *
     * Notice: we pass vec (not &vec)
     * - vec is already a pointer to the first element.
     * - MPI_Bcast expects the address of the buffer containing elements.
     */
    MPI_Bcast(vec, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* =========================== Phase 6: Load/scatter matrix =================== */

    /*
     * Rank 0 loads the full dim×dim matrix.
     * Other ranks do not need the full matrix, only their row block.
     */
    if (prank == 0)
        tmat = loadMat(mfname, dim);

    /*
     * Compute how many matrix elements each process will receive.
     *
     * We split by rows, equally:
     * - each rank gets (dim / csize) rows
     * - each row has dim columns
     * -> elements per rank = dim * (dim / csize)
     *
     * Because of integer division:
     * - dim/csize must be an integer with no remainder for correct equal partitioning.
     */
    int msize = dim * dim / csize;

    /*
     * Allocate local matrix chunk.
     * This will store exactly msize doubles received from MPI_Scatter.
     */
    mat = new double[msize];

    /*
     * MPI_Scatter:
     * - Root (rank 0) sends a distinct chunk of size msize to each rank.
     * - Each rank receives exactly one chunk into its local `mat`.
     *
     * Parameters (send side):
     * - tmat: root send buffer (only meaningful on rank 0)
     * - msize: number of elements sent to each rank
     * - MPI_DOUBLE: datatype
     *
     * Parameters (receive side):
     * - mat: local receive buffer on each rank
     * - msize: number of elements received
     * - MPI_DOUBLE: datatype
     *
     * Collective rule:
     * - All ranks must call MPI_Scatter.
     */
    MPI_Scatter(
        tmat, msize, MPI_DOUBLE,   /* send buffer (root only) */
        mat,  msize, MPI_DOUBLE,   /* receive buffer (all ranks) */
        0, MPI_COMM_WORLD
    );

    /* =================== Phase 7: Local computation (matrix-vector multiply) ==== */

    /*
     * int to = dim / csize;
     *
     * `to` is the number of rows assigned to each rank.
     * (Think: “how many output elements this rank will produce”.)
     *
     * Again, requires dim divisible by csize for equal partitioning.
     */
    int to = dim / csize;

    /*
     * Allocate local result vector for this rank.
     * Each rank produces `to` outputs corresponding to its `to` rows.
     */
    lres = new double[to];

    /*
     * Local multiplication:
     *
     * mat is stored as `to` consecutive rows, each of length dim:
     *   local row i has elements: mat[i*dim + j], j=0..dim-1
     *
     * vec is the full vector, available on every rank.
     *
     * For each local row i:
     *   lres[i] = Σ_j ( mat[i*dim + j] * vec[j] )
     *
     * This is a dot product.
     */
    for (int i = 0; i != to; ++i)
    {
        /*
         * double s = 0;
         * - accumulator for the dot product of row i with the vector.
         */
        double s = 0;

        /*
         * for each column j, accumulate mat[i,j] * vec[j]
         */
        for (int j = 0; j != dim; ++j)
            s += mat[i * dim + j] * vec[j];

        /*
         * store the computed output entry for local row i
         */
        lres[i] = s;
    }

    /* =================== Phase 8: Gather results back to rank 0 ================= */

    /*
     * Rank 0 allocates space for the complete result vector of length dim.
     * Other ranks do not need the full result, only their local lres.
     */
    if (prank == 0)
        res = new double[dim];

    /*
     * MPI_Gather:
     * - Each rank sends its `to` result elements.
     * - Root rank receives them into `res` in rank order:
     *     res[0..to-1]         from rank 0
     *     res[to..2*to-1]      from rank 1
     *     ...
     *
     * This matches the row-block distribution if the scatter also used rank order.
     */
    MPI_Gather(
        lres, to, MPI_DOUBLE,   /* send buffer on each rank */
        res,  to, MPI_DOUBLE,   /* receive buffer on root */
        0, MPI_COMM_WORLD
    );

    /* =================== Phase 9: Output result file (rank 0 only) ============== */

    if (prank == 0)
    {
        /*
         * Write the full result vector to a file.
         * Only rank 0 does this to avoid multiple ranks overwriting the same file.
         */
        logRes("Result.txt", res, dim);
    }

    /* =================== Phase 10: Cleanup (free memory) ======================== */

    /*
     * Memory ownership summary:
     * - vec: allocated on all ranks, must be deleted on all ranks.
     * - mat: allocated on all ranks, must be deleted on all ranks.
     * - lres: allocated on all ranks, must be deleted on all ranks.
     * - tmat: allocated only on rank 0, delete only on rank 0.
     * - res : allocated only on rank 0, delete only on rank 0.
     */
    if (prank == 0)
    {
        delete[] tmat;
        delete[] res;
    }

    delete[] vec;
    delete[] mat;
    delete[] lres;

    /* =================== Phase 11: Finalize MPI ================================= */

    /*
     * MPI_Finalize:
     * - Clean shutdown of the MPI runtime.
     * - No MPI calls after this point (except a few special cases).
     */
    MPI_Finalize();

    return 0;
}
