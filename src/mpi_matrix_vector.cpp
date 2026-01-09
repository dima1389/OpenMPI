/*
 * File:
 *   mpi_matvec_scatter_gather.c
 *
 * Purpose:
 *   Demonstrate MPI collective communication for distributed matrix-vector multiplication.
 *
 * Description:
 *   This example demonstrates parallel matrix-vector multiplication (y = A * x) using:
 *     - MPI_Bcast   to distribute problem size and the full input vector to all ranks,
 *     - MPI_Scatter to distribute contiguous row blocks of the matrix to ranks,
 *     - MPI_Gather  to collect partial result blocks back to rank 0.
 *
 *   Observable outcome:
 *     - Rank 0 writes the resulting vector to "Result.txt".
 *
 * Key concepts:
 *   - ranks, SPMD execution, collective communication
 *   - deterministic data distribution via row-block partitioning
 *   - imbalance risk if work is not uniform (not the case here for dense uniform rows)
 *
 * Algorithm / workflow (high level):
 *   1) Rank 0 reads vector length (dim) from file; broadcast dim.
 *   2) Rank 0 loads full vector; broadcast vector to all ranks.
 *   3) Rank 0 loads full matrix; scatter equal row blocks to ranks.
 *   4) Each rank computes its local partial result for its row block.
 *   5) Gather partial results to rank 0; rank 0 writes output.
 *
 * MPI features used:
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size, MPI_Bcast, MPI_Scatter, MPI_Gather, MPI_Abort
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_matvec_scatter_gather.c -o mpi_matvec_scatter_gather
 *
 * Execution:
 *   mpiexec -n <P> mpi_matvec_scatter_gather <vector_file> <matrix_file>
 *
 * Inputs:
 *   - Command-line arguments:
 *       argv[1] = path to vector file (one double per entry)
 *       argv[2] = path to matrix file (dim*dim doubles, row-major)
 *
 * Constraints:
 *   - dim must be divisible by the number of processes (csize) for equal row-block partitioning.
 *
 * References:
 *   - MPI Standard: Collective Communication (Broadcast, Scatter, Gather)
 */

#include <stdio.h>   // FILE, fopen, fscanf, fprintf, fclose, stderr, fprintf
#include <stdlib.h>  // malloc, free, EXIT_FAILURE, EXIT_SUCCESS
#include <mpi.h>     // MPI API

// -----------------------------------------------------------------------------
// returnSize
// -----------------------------------------------------------------------------
// Reads a text file with one double per entry and returns how many numbers
// are stored in that file.
//
// Parameters:
//   fname - path to the text file with double values
//
// Returns:
//   Number of doubles stored in the file (dimension of the vector).
// -----------------------------------------------------------------------------
static int returnSize(const char *fname)                                  // Define helper to count doubles in a file to determine vector dimension.
{
    FILE *f = fopen(fname, "r");                                          // Open the input file for reading to scan double values.
    if (f == NULL) {                                                      // Validate file handle to avoid undefined behavior on fscanf/fclose.
        return -1;                                                        // Return a sentinel error value so caller can abort collectively.
    }

    int dim = 0;                                                          // Initialize counter of successfully read doubles.
    double tmp = 0.0;                                                     // Provide storage for fscanf to parse each double token.

    while (fscanf(f, "%lf", &tmp) == 1) {                                 // Read doubles until parsing fails, counting only successful conversions.
        dim++;                                                            // Increment dimension for each parsed element.
    }

    fclose(f);                                                            // Close file to release OS resources and flush buffers (if any).
    return dim;                                                           // Return the determined vector length.
}

// -----------------------------------------------------------------------------
// loadVec
// -----------------------------------------------------------------------------
// Allocates and loads a vector (1D array) of size n from a text file.
//
// Assumes the file has at least n double values separated by whitespace.
//
// Parameters:
//   fname - path to the file with vector elements
//   n     - expected number of elements
//
// Returns:
//   Pointer to a dynamically allocated array of n doubles.
//   Caller is responsible for free().
// -----------------------------------------------------------------------------
static double *loadVec(const char *fname, int n)                           // Define helper to allocate and populate a vector from a text file.
{
    FILE *f = fopen(fname, "r");                                          // Open vector file for reading to load n doubles.
    if (f == NULL) {                                                      // Validate file handle to prevent dereferencing NULL in fscanf/fclose.
        return NULL;                                                      // Return NULL so caller can handle the error consistently across ranks.
    }

    double *res = (double *)malloc((size_t)n * sizeof(double));           // Allocate contiguous memory for n doubles in C style.
    if (res == NULL) {                                                    // Validate allocation to avoid writes through a NULL pointer.
        fclose(f);                                                        // Close file before returning to avoid leaking file descriptor.
        return NULL;                                                      // Propagate allocation failure to caller for collective abort.
    }

    for (int i = 0; i < n; i++) {                                         // Iterate exactly n times to enforce expected vector length.
        if (fscanf(f, "%lf", &res[i]) != 1) {                             // Read one double per element and detect premature EOF/format errors.
            free(res);                                                    // Free partially allocated buffer to avoid a heap leak on error.
            fclose(f);                                                    // Close file handle before returning to maintain resource hygiene.
            return NULL;                                                  // Signal load failure so caller can abort the MPI job deterministically.
        }
    }

    fclose(f);                                                            // Close the input file after successful reads to release resources.
    return res;                                                           // Return populated vector buffer to caller.
}

// -----------------------------------------------------------------------------
// loadMat
// -----------------------------------------------------------------------------
// Allocates and loads a matrix of size n x n from a text file.
//
// The matrix is stored in a 1D array in row-major order:
//
//   res[ i * n + j ] = element at row i, column j
//
// Assumes the file has at least n*n double values.
//
// Parameters:
//   fname - path to the file with matrix elements
//   n     - dimension of the matrix (n x n)
//
// Returns:
//   Pointer to a dynamically allocated array of n*n doubles.
//   Caller is responsible for free().
// -----------------------------------------------------------------------------
static double *loadMat(const char *fname, int n)                           // Define helper to allocate and populate an n-by-n matrix from text.
{
    FILE *f = fopen(fname, "r");                                          // Open matrix file for reading to load n*n doubles.
    if (f == NULL) {                                                      // Validate file handle to avoid invalid I/O calls.
        return NULL;                                                      // Return NULL so caller can handle consistently in MPI context.
    }

    size_t count = (size_t)n * (size_t)n;                                 // Compute total number of matrix elements with size_t to avoid overflow in loops.
    double *res = (double *)malloc(count * sizeof(double));               // Allocate contiguous row-major storage for the full matrix.
    if (res == NULL) {                                                    // Validate allocation to prevent undefined behavior on writes.
        fclose(f);                                                        // Close file handle before returning to prevent resource leak.
        return NULL;                                                      // Signal allocation failure to the caller.
    }

    for (size_t k = 0; k < count; k++) {                                  // Iterate over all matrix elements in row-major linear index.
        if (fscanf(f, "%lf", &res[k]) != 1) {                             // Read each required double and detect missing/invalid tokens.
            free(res);                                                    // Free buffer to avoid leaking heap memory on partial read failure.
            fclose(f);                                                    // Close file before returning to maintain correct resource lifecycle.
            return NULL;                                                  // Propagate failure to caller for MPI-wide abort handling.
        }
    }

    fclose(f);                                                            // Close file after successful read to release OS resources.
    return res;                                                           // Return populated matrix buffer (row-major) to caller.
}

// -----------------------------------------------------------------------------
// logRes
// -----------------------------------------------------------------------------
// Writes the result vector to a text file, one line with all values
// separated by spaces.
//
// Parameters:
//   fname - output file name
//   res   - pointer to result vector
//   n     - length of result vector
// -----------------------------------------------------------------------------
static void logRes(const char *fname, const double *res, int n)            // Define helper to write a vector result deterministically to disk.
{
    FILE *f = fopen(fname, "w");                                          // Open output file for writing, truncating any existing content.
    if (f == NULL) {                                                      // Validate output file handle to avoid fprintf on NULL.
        return;                                                           // Return without writing since caller may still finalize MPI cleanly.
    }

    for (int i = 0; i < n; i++) {                                         // Iterate over all n result elements to serialize them to file.
        fprintf(f, "%lf ", res[i]);                                       // Emit each element with a space delimiter for simple parsing.
    }

    fclose(f);                                                            // Close output file to flush buffers and finalize the file on disk.
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
// MPI parallel matrix-vector multiplication.
//
// Input arguments (command line):
//   argv[1] - path to vector file (vfname)
//   argv[2] - path to matrix file (mfname)
//
// Vector length = dim
// Matrix size   = dim x dim (stored in row-major order in the file)
//
// The work is divided by rows of the matrix across MPI processes.
// It assumes that dim is divisible by number of processes (csize).
// -----------------------------------------------------------------------------
int main(int argc, char *argv[])                                           // Entry point for all MPI ranks executing the same SPMD program.
{
    int csize = 0;                                                         // Declare communicator size to determine partitioning of matrix rows.
    int prank = 0;                                                         // Declare rank to control root-only I/O and collective root behavior.

    MPI_Init(&argc, &argv);                                                // Initialize MPI runtime; required before any other MPI call.
    MPI_Comm_size(MPI_COMM_WORLD, &csize);                                 // Query number of ranks to compute per-rank work and message sizes.
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);                                 // Query rank ID to select root (rank 0) responsibilities.

    if (argc < 3) {                                                        // Validate command-line inputs to avoid out-of-bounds argv access.
        if (prank == 0) {                                                  // Restrict usage message to root to avoid duplicated output.
            fprintf(stderr, "Usage: %s <vector_file> <matrix_file>\n", argv[0]); // Print correct invocation format for reproducibility.
        }
        MPI_Abort(MPI_COMM_WORLD, 1);                                      // Abort the MPI job collectively since the program cannot proceed.
    }

    const char *vfname = argv[1];                                          // Capture vector file path from argv for consistent use.
    const char *mfname = argv[2];                                          // Capture matrix file path from argv for consistent use.

    int dim = 0;                                                           // Declare global dimension (vector length / matrix dimension).
    double *tmat = NULL;                                                   // Hold full matrix on root only to serve as MPI_Scatter send buffer.
    double *vec = NULL;                                                    // Hold full vector on all ranks after broadcast.
    double *mat = NULL;                                                    // Hold local matrix row-block (scatter receive buffer) on each rank.
    double *lres = NULL;                                                   // Hold local result block for this rank's rows.
    double *res = NULL;                                                    // Hold full result on root only (gather receive buffer).

    if (prank == 0) {                                                      // Restrict dimension detection to root to avoid redundant file I/O.
        dim = returnSize(vfname);                                          // Count number of doubles in the vector file to determine dim.
        if (dim <= 0) {                                                    // Validate dimension to ensure downstream allocations and partitioning are valid.
            fprintf(stderr, "Error: failed to read vector dimension from '%s'\n", vfname); // Report root-side I/O failure clearly.
            MPI_Abort(MPI_COMM_WORLD, 2);                                  // Abort collectively since all ranks depend on dim.
        }
    }

    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);                        // Broadcast dimension so every rank allocates consistent buffers.

    if (dim % csize != 0) {                                                // Enforce equal row-block partitioning assumption for Scatter/Gather counts.
        if (prank == 0) {                                                  // Emit constraint violation only once to keep output readable.
            fprintf(stderr, "Error: dim=%d is not divisible by csize=%d\n", dim, csize); // Explain why the run cannot proceed.
        }
        MPI_Abort(MPI_COMM_WORLD, 3);                                      // Abort all ranks to avoid mismatched message sizes and incorrect results.
    }

    if (prank == 0) {                                                      // Restrict vector loading to root to minimize file I/O and duplication.
        vec = loadVec(vfname, dim);                                        // Load full vector so it can be broadcast to all ranks.
        if (vec == NULL) {                                                 // Validate vector load to ensure broadcast source is valid.
            fprintf(stderr, "Error: failed to load vector from '%s'\n", vfname); // Report specific file that failed to load.
            MPI_Abort(MPI_COMM_WORLD, 4);                                  // Abort because computation cannot proceed without the input vector.
        }
    } else {
        vec = (double *)malloc((size_t)dim * sizeof(double));              // Allocate receive buffer for broadcasted vector on non-root ranks.
        if (vec == NULL) {                                                 // Validate allocation to prevent MPI_Bcast writing to NULL.
            fprintf(stderr, "Error: rank %d failed to allocate vector buffer\n", prank); // Provide rank-specific diagnostic.
            MPI_Abort(MPI_COMM_WORLD, 5);                                  // Abort to prevent undefined behavior during the broadcast.
        }
    }

    MPI_Bcast(vec, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);                    // Broadcast full vector so each rank can compute its local dot-products.

    if (prank == 0) {                                                      // Restrict full matrix loading to root since only root scatters it.
        tmat = loadMat(mfname, dim);                                       // Load full dim-by-dim matrix in row-major order to scatter by row blocks.
        if (tmat == NULL) {                                                // Validate matrix load to ensure MPI_Scatter send buffer is valid.
            fprintf(stderr, "Error: failed to load matrix from '%s'\n", mfname); // Report matrix file load failure for debugging.
            MPI_Abort(MPI_COMM_WORLD, 6);                                  // Abort since all ranks depend on the matrix distribution.
        }
    }

    int rows_per_rank = dim / csize;                                       // Compute equal number of rows assigned to each rank by construction.
    int msize = rows_per_rank * dim;                                       // Compute local matrix element count (rows_per_rank rows, dim columns).

    mat = (double *)malloc((size_t)msize * sizeof(double));                // Allocate receive buffer for this rank's contiguous row block.
    if (mat == NULL) {                                                     // Validate allocation before calling MPI_Scatter into mat.
        fprintf(stderr, "Error: rank %d failed to allocate local matrix buffer\n", prank); // Provide rank-local allocation diagnostics.
        MPI_Abort(MPI_COMM_WORLD, 7);                                      // Abort since scatter receive buffer is required on all ranks.
    }

    MPI_Scatter(                                                           // Distribute equal contiguous row blocks of the matrix from root to all ranks.
        tmat, msize, MPI_DOUBLE,                                           // Send msize doubles per rank from root's full matrix buffer.
        mat,  msize, MPI_DOUBLE,                                           // Receive msize doubles into each rank's local matrix buffer.
        0, MPI_COMM_WORLD                                                  // Use rank 0 as the root of the scatter on MPI_COMM_WORLD.
    );

    lres = (double *)malloc((size_t)rows_per_rank * sizeof(double));        // Allocate local result block matching this rank's assigned rows.
    if (lres == NULL) {                                                    // Validate allocation to avoid writes to NULL during local computation.
        fprintf(stderr, "Error: rank %d failed to allocate local result buffer\n", prank); // Report rank-specific allocation error.
        MPI_Abort(MPI_COMM_WORLD, 8);                                      // Abort since gathering requires valid send buffers.
    }

    for (int i = 0; i < rows_per_rank; i++) {                              // Iterate over each local row to compute one output element per row.
        double s = 0.0;                                                    // Initialize accumulator for the dot-product of row i with vector x.
        for (int j = 0; j < dim; j++) {                                    // Iterate over all columns to compute dense dot-product.
            s += mat[i * dim + j] * vec[j];                                // Accumulate A_local(i,j) * x(j) into the row's dot-product sum.
        }
        lres[i] = s;                                                       // Store computed output element for this local row index.
    }

    if (prank == 0) {                                                      // Restrict allocation of final result buffer to root (gather destination).
        res = (double *)malloc((size_t)dim * sizeof(double));              // Allocate full output vector to receive gathered blocks from all ranks.
        if (res == NULL) {                                                 // Validate allocation to ensure MPI_Gather receive buffer is valid.
            fprintf(stderr, "Error: rank 0 failed to allocate final result buffer\n"); // Report root-side allocation failure.
            MPI_Abort(MPI_COMM_WORLD, 9);                                  // Abort since gather cannot complete without a valid receive buffer.
        }
    }

    MPI_Gather(                                                            // Collect all local result blocks to form the full output vector on root.
        lres, rows_per_rank, MPI_DOUBLE,                                   // Send each rank's rows_per_rank outputs to root.
        res,  rows_per_rank, MPI_DOUBLE,                                   // Receive concatenated blocks into root's res buffer in rank order.
        0, MPI_COMM_WORLD                                                  // Use rank 0 as the gather root so it can write the output file.
    );

    if (prank == 0) {                                                      // Restrict output logging to root to avoid concurrent file writes.
        logRes("Result.txt", res, dim);                                    // Write the computed result vector to disk for inspection and grading.
    }

    if (prank == 0) {                                                      // Restrict root-only frees to pointers that were allocated only on root.
        free(tmat);                                                        // Free full matrix buffer after scatter since it is no longer needed.
        free(res);                                                         // Free final result buffer after logging to avoid heap leaks.
    }

    free(vec);                                                             // Free broadcasted vector buffer on all ranks to release heap memory.
    free(mat);                                                             // Free local matrix block on all ranks after computation completes.
    free(lres);                                                            // Free local result block on all ranks after gather completes.

    MPI_Finalize();                                                        // Finalize MPI runtime to cleanly tear down communication resources.

    return EXIT_SUCCESS;                                                   // Return success code to the host environment after normal completion.
}
