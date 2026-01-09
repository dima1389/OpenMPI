/*
 * File:
 *   mpi_matrix_vector_scatterv_gatherv.c
 *
 * Purpose:
 *   Demonstrate distributed dense matrix-vector multiplication (y = A * x) using uneven row-block partitioning.
 *
 * Description:
 *   This example computes y = A * x for a dense n×n matrix A and length-n vector x, where n is inferred from
 *   the vector file. Rows of A are distributed across ranks using MPI_Scatterv so that n does not need to be
 *   divisible by the number of processes p; each rank computes its local y segment and rank 0 collects the full
 *   y using MPI_Gatherv and writes Result.txt.
 *
 * Key concepts:
 *   - ranks, communicators, collectives
 *   - deterministic, blocking behavior
 *   - imbalance handling via uneven distribution (q = n/p, r = n%p)
 *
 * Algorithm / workflow (high level):
 *   1) Rank 0 determines n from the vector file and broadcasts n
 *   2) Rank 0 computes Scatterv/Gatherv counts and displacements for uneven row blocks
 *   3) Rank 0 loads x and broadcasts x to all ranks
 *   4) Rank 0 loads A and scatters row blocks of A; each rank computes its partial y
 *   5) Rank 0 gathers partial y blocks, writes Result.txt, and all ranks clean up and finalize
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize
 *   - MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Scatterv, MPI_Gatherv
 *   - MPI_Abort
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_matrix_vector_scatterv_gatherv.c -o mpi_matrix_vector_scatterv_gatherv
 *
 * Execution:
 *   mpiexec -n <P> mpi_matrix_vector_scatterv_gatherv <vector_file> <matrix_file>
 *
 * Inputs:
 *   - Command-line arguments: <vector_file> <matrix_file>
 *   - Interactive input: none
 *
 * Output:
 *   - Rank 0 writes: Result.txt (n doubles, space-separated)
 *
 * References:
 *   - MPI Standard: Collective communication (MPI_Bcast, MPI_Scatterv, MPI_Gatherv), Process termination (MPI_Abort)
 */

#include <stdio.h>   // Provides FILE, fopen/fclose, fscanf/fprintf, stderr, printf for file and console I/O.
#include <stdlib.h>  // Provides malloc/free for dynamic memory management and size_t definitions.
#include <mpi.h>     // Provides MPI types and functions for initializing, communicating, and finalizing MPI programs.

/*
 * Generalized dense matrix-vector multiplication: y = A * x
 *
 * Key feature:
 *  - Works for any n and any number of processes (n does NOT need to be a multiple of p).
 *  - Uses uneven row-block distribution with MPI_Scatterv / MPI_Gatherv.
 *
 * Input format (whitespace separated doubles):
 *  - Vector file: n doubles
 *  - Matrix file: n*n doubles in row-major order
 *
 * Usage:
 *   mpiexec -n <p> MPI_Matrix_Vector_General <vector_file> <matrix_file>
 *
 * Output (rank 0):
 *   Result.txt containing n doubles (space-separated)
 */

static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)              // Defines a rank-aware fatal-error helper that aborts the communicator.
{
    if (rank == 0) {                                                              // Restricts user-visible error reporting to rank 0 to avoid duplicated messages.
        fprintf(stderr, "ERROR: %s\n", msg);                                      // Prints the diagnostic to stderr so it is not mixed with normal program output.
    }
    MPI_Abort(comm, 1);                                                           // Terminates all ranks in the communicator to prevent deadlock after a fatal error.
}

/* Count how many doubles are present in a file (vector size). */
static int count_doubles_in_file(const char *fname)                               // Declares a utility that infers vector length by counting parsable doubles in a file.
{
    FILE *f = fopen(fname, "r");                                                  // Opens the input file in text read mode to scan whitespace-separated doubles.
    if (!f) return -1;                                                            // Signals failure to open the file so the caller can handle missing/unreadable input.

    int count = 0;                                                                // Initializes the parsed-double counter used to derive n.
    double tmp;                                                                   // Declares a temporary storage location for fscanf so values need not be retained.
    while (fscanf(f, "%lf", &tmp) == 1) {                                         // Repeatedly scans doubles until input is exhausted or a parse error occurs.
        count++;                                                                  // Increments for each successfully scanned double to compute the file’s numeric length.
    }
    fclose(f);                                                                    // Closes the file to release the OS handle and flush any buffered state.
    return count;                                                                 // Returns the inferred length so rank 0 can define the global problem size n.
}

static double *load_vector(const char *fname, int n)                              // Declares a loader that reads exactly n doubles into a newly allocated vector.
{
    FILE *f = fopen(fname, "r");                                                  // Opens the vector file for sequential scanning.
    if (!f) return NULL;                                                          // Returns NULL so callers can detect and report I/O failures.

    double *x = (double *)malloc((size_t)n * sizeof(double));                     // Allocates contiguous storage for n doubles so x can be broadcast efficiently.
    if (!x) { fclose(f); return NULL; }                                           // Handles allocation failure and ensures the file handle is not leaked.

    for (int i = 0; i < n; i++) {                                                 // Iterates exactly n times to enforce a strict input size contract.
        if (fscanf(f, "%lf", &x[i]) != 1) {                                       // Validates that each expected element exists and matches the double format.
            free(x);                                                              // Frees the partially filled buffer to avoid memory leaks on parse failure.
            fclose(f);                                                            // Closes the file before returning to maintain resource correctness.
            return NULL;                                                          // Signals format/size mismatch to the caller for consistent error handling.
        }
    }
    fclose(f);                                                                    // Closes the vector file after successful loading.
    return x;                                                                     // Returns the populated vector so rank 0 can copy and broadcast it.
}

static double *load_matrix(const char *fname, int n)                              // Declares a loader that reads an n×n row-major matrix into a newly allocated array.
{
    FILE *f = fopen(fname, "r");                                                  // Opens the matrix file for sequential scanning.
    if (!f) return NULL;                                                          // Returns NULL so callers can distinguish open failures from parse failures.

    size_t m = (size_t)n * (size_t)n;                                             // Computes the total element count using size_t to avoid signed overflow in indexing.
    double *A = (double *)malloc(m * sizeof(double));                             // Allocates a contiguous buffer to enable Scatterv of row blocks.
    if (!A) { fclose(f); return NULL; }                                           // Handles allocation failure while preserving file-handle hygiene.

    for (size_t i = 0; i < m; i++) {                                              // Scans exactly m doubles to match the required matrix size.
        if (fscanf(f, "%lf", &A[i]) != 1) {                                       // Validates each expected value exists and is parseable as a double.
            free(A);                                                              // Frees partially read matrix storage to avoid leaks.
            fclose(f);                                                            // Closes the file before error return to release OS resources.
            return NULL;                                                          // Signals invalid format or insufficient data to the caller.
        }
    }
    fclose(f);                                                                    // Closes the matrix file after successful loading.
    return A;                                                                     // Returns the full matrix buffer so rank 0 can scatter row blocks.
}

static void write_result(const char *fname, const double *y, int n)               // Declares an output routine that writes n doubles to a file in a simple text format.
{
    FILE *f = fopen(fname, "w");                                                  // Opens/creates the output file in write mode to emit the result vector.
    if (!f) return;                                                               // Silently returns on open failure to avoid crashing during non-essential output.

    for (int i = 0; i < n; i++) {                                                 // Iterates over all entries to serialize the full result vector.
        fprintf(f, "%lf%s", y[i], (i + 1 == n) ? "" : " ");                       // Writes each value with a space delimiter while avoiding a trailing space.
    }
    fprintf(f, "\n");                                                             // Terminates the line to make the output file line-oriented and user-friendly.
    fclose(f);                                                                    // Closes the output file to flush buffered output and release the OS handle.
}

int main(int argc, char **argv)                                                   // Defines the MPI program entry point, receiving standard command-line arguments.
{
    MPI_Init(&argc, &argv);                                                       // Initializes MPI so collective operations and communicator queries are valid.

    int rank, p;                                                                  // Declares process rank and world size for partitioning and rank-specific control flow.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);                                         // Retrieves this process’s rank within MPI_COMM_WORLD for role selection.
    MPI_Comm_size(MPI_COMM_WORLD, &p);                                            // Retrieves the total number of ranks to compute the row distribution.

    if (argc != 3) {                                                              // Validates that both required file arguments were provided by the user.
        if (rank == 0) {                                                          // Restricts usage printing to rank 0 to avoid redundant messages.
            fprintf(stderr, "Usage: %s <vector_file> <matrix_file>\n", argv[0]);  // Reports the correct invocation syntax to stderr for visibility.
        }
        MPI_Finalize();                                                           // Finalizes MPI to shut down the runtime cleanly on early exit.
        return 1;                                                                 // Returns a nonzero status code to indicate incorrect usage to the OS.
    }

    const char *vec_file = argv[1];                                               // Stores the vector file path for consistent use across helper calls.
    const char *mat_file = argv[2];                                               // Stores the matrix file path for consistent use across helper calls.

    int n = 0;                                                                    // Initializes the global dimension n so it can be broadcast after rank 0 computes it.

    /* Rank 0 determines n from the vector file. */
    if (rank == 0) {                                                              // Ensures only rank 0 performs file scanning to avoid duplicated I/O and inconsistency.
        n = count_doubles_in_file(vec_file);                                      // Infers n by counting doubles in the vector file, defining the problem size.
        if (n <= 0) {                                                             // Rejects empty or unreadable input to prevent undefined distribution and allocation sizes.
            fprintf(stderr, "ERROR: cannot read vector size from file '%s'\n", vec_file); // Prints an explicit diagnostic to support rapid user remediation.
            MPI_Abort(MPI_COMM_WORLD, 1);                                         // Aborts all ranks to prevent deadlock due to inconsistent control flow.
        }
    }

    /* Broadcast n to all ranks. */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);                                 // Distributes n so every rank can allocate buffers and compute offsets consistently.

    /* Compute uneven row distribution: rows_i and offset_i for each rank i. */
    int q = n / p;                                                                // Computes the base number of rows per rank under integer division.
    int r = n % p;                                                                // Computes the remainder rows to be distributed one-per-rank among the first r ranks.

    int local_rows = q + (rank < r ? 1 : 0);                                      // Assigns this rank either q or q+1 rows to achieve an uneven but complete partition.
    int local_row_offset = rank * q + (rank < r ? rank : r);                      // Computes this rank’s starting global row index with a correction for distributed remainders.

    /* Prepare counts/displacements for Scatterv/Gatherv (rank 0 needs full arrays). */
    int *sendcountsA = NULL;                                                      // Declares rank-0-only send counts for scattering matrix row blocks.
    int *displsA     = NULL;                                                      // Declares rank-0-only displacements (in elements) for matrix scattering.
    int *recvcountsY = NULL;                                                      // Declares rank-0-only receive counts for gathering result segments.
    int *displsY     = NULL;                                                      // Declares rank-0-only displacements (in elements) for result gathering.

    if (rank == 0) {                                                              // Allocates and populates metadata arrays only on rank 0 as required by MPI_Scatterv/Gatherv.
        sendcountsA = (int *)malloc((size_t)p * sizeof(int));                     // Allocates per-rank counts for the number of matrix elements to send.
        displsA     = (int *)malloc((size_t)p * sizeof(int));                     // Allocates per-rank displacements into Afull for Scatterv.
        recvcountsY = (int *)malloc((size_t)p * sizeof(int));                     // Allocates per-rank counts for the number of y elements to receive.
        displsY     = (int *)malloc((size_t)p * sizeof(int));                     // Allocates per-rank displacements into y for Gatherv.

        if (!sendcountsA || !displsA || !recvcountsY || !displsY) {               // Checks for allocation failures that would invalidate collective argument arrays.
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for counts/displacements"); // Aborts consistently if metadata cannot be allocated.
        }

        int dispA = 0;                                                            // Initializes the running matrix displacement in elements for building displsA.
        int dispY = 0;                                                            // Initializes the running result displacement in elements for building displsY.
        for (int i = 0; i < p; i++) {                                             // Iterates over all ranks to compute their row counts and offsets.
            int rows_i = q + (i < r ? 1 : 0);                                     // Computes rank i’s row ownership using the same uneven distribution rule.

            sendcountsA[i] = rows_i * n;                                          // Sets the number of matrix elements for rank i (rows_i rows × n columns).
            displsA[i]     = dispA;                                               // Records the starting element index within Afull for rank i’s block.

            recvcountsY[i] = rows_i;                                              // Sets the number of result elements contributed by rank i (one per owned row).
            displsY[i]     = dispY;                                               // Records the starting element index within y where rank i’s segment is placed.

            dispA += sendcountsA[i];                                              // Advances the matrix displacement by rank i’s block size for the next rank.
            dispY += recvcountsY[i];                                              // Advances the result displacement by rank i’s segment size for the next rank.
        }
    }

    /* Allocate and load x (broadcast to all). */
    double *x = (double *)malloc((size_t)n * sizeof(double));                     // Allocates the full vector x on every rank to support local dot-products.
    if (!x) {                                                                     // Validates allocation since x is required for all ranks’ computations.
        die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for vector x");      // Aborts the job to avoid undefined behavior from NULL dereferences.
    }

    if (rank == 0) {                                                              // Ensures only rank 0 reads the vector file to avoid filesystem contention.
        double *tmp = load_vector(vec_file, n);                                   // Loads x from file into a temporary buffer to validate the input format strictly.
        if (!tmp) {                                                               // Detects file open or parse failures for the required n values.
            free(x);                                                              // Frees the per-rank allocation before aborting to keep cleanup correct on rank 0.
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read vector file (format/size mismatch)"); // Aborts with a precise failure reason.
        }
        for (int i = 0; i < n; i++) x[i] = tmp[i];                                // Copies validated values into the broadcast buffer used by all ranks.
        free(tmp);                                                                // Frees the temporary buffer since x now owns the finalized vector data.
    }

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);                               // Broadcasts x so every rank can compute y rows without further communication.

    /* Rank 0 loads full matrix A; others keep NULL. */
    double *Afull = NULL;                                                         // Declares the full matrix pointer, valid only on rank 0 for scattering.
    if (rank == 0) {                                                              // Limits matrix file I/O to rank 0 to centralize parsing and distribution.
        Afull = load_matrix(mat_file, n);                                         // Loads n×n doubles in row-major order as required by the distribution scheme.
        if (!Afull) {                                                             // Detects file open or parse failures for the required n*n values.
            free(x);                                                              // Frees x to avoid leaking rank-0 allocations in the fatal path.
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read matrix file (format/size mismatch)"); // Aborts with a format/size diagnostic.
        }
    }

    /* Allocate local matrix chunk: local_rows * n */
    double *Alocal = NULL;                                                        // Declares the per-rank matrix slice pointer received via MPI_Scatterv.
    if (local_rows > 0) {                                                         // Avoids allocating zero-length buffers on ranks assigned no rows.
        Alocal = (double *)malloc((size_t)local_rows * (size_t)n * sizeof(double)); // Allocates space for this rank’s contiguous block of rows.
        if (!Alocal) {                                                            // Checks allocation because Alocal is required for local computation.
            free(x);                                                              // Frees x since it was allocated on all ranks and must not leak on failure.
            if (rank == 0) free(Afull);                                           // Frees Afull on rank 0 to avoid leaking the full matrix on abort.
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local matrix chunk"); // Aborts since computation cannot proceed without local rows.
        }
    }

    /* Scatter uneven row blocks of A. */
    MPI_Scatterv(                                                                 // Starts a variable-count scatter to distribute row blocks that may differ by rank.
        Afull, sendcountsA, displsA, MPI_DOUBLE,                                  // Specifies rank-0 send buffer, per-rank element counts, and element displacements.
        Alocal, local_rows * n, MPI_DOUBLE,                                       // Specifies per-rank receive buffer sized to exactly local_rows*n elements.
        0, MPI_COMM_WORLD                                                         // Sets rank 0 as the root and MPI_COMM_WORLD as the collective communicator.
    );

    /* Compute local result y_local = A_local * x */
    double *ylocal = NULL;                                                        // Declares the per-rank output buffer for this rank’s computed y segment.
    if (local_rows > 0) {                                                         // Skips allocation and computation on ranks that own no rows.
        ylocal = (double *)malloc((size_t)local_rows * sizeof(double));           // Allocates one output value per owned row to be gathered later.
        if (!ylocal) {                                                            // Checks allocation because ylocal is required to participate in MPI_Gatherv correctly.
            free(x);                                                              // Frees x to avoid memory leaks along the abort path.
            if (rank == 0) free(Afull);                                           // Frees Afull if present to maintain rank-0 cleanup symmetry.
            free(Alocal);                                                         // Frees the already allocated local matrix slice to avoid leaking received data.
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local result chunk"); // Aborts since results cannot be computed or gathered safely.
        }

        for (int i = 0; i < local_rows; i++) {                                    // Iterates over each owned row to compute its dot-product with x.
            double sum = 0.0;                                                     // Initializes the accumulator to implement a standard inner product.
            const double *row = &Alocal[(size_t)i * (size_t)n];                   // Computes the base address of row i within the contiguous row-block layout.
            for (int j = 0; j < n; j++) {                                         // Iterates over all columns to complete the dot-product for row i.
                sum += row[j] * x[j];                                             // Accumulates A[i,j]*x[j] to build y[i] as required by y = A*x.
            }
            ylocal[i] = sum;                                                      // Stores the computed dot-product so it can be gathered to rank 0.
        }
    }

    /* Gather uneven y chunks to rank 0. */
    double *y = NULL;                                                             // Declares the full output vector pointer, allocated only on rank 0.
    if (rank == 0) {                                                              // Allocates the global result only where it will be written to disk.
        y = (double *)malloc((size_t)n * sizeof(double));                         // Allocates storage for the full result vector y of length n.
        if (!y) {                                                                 // Checks allocation because rank 0 must receive all gathered results.
            free(x);                                                              // Frees x on rank 0 before abort to keep memory accounting correct.
            free(Afull);                                                          // Frees Afull because it is no longer needed if allocation fails.
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for full result y"); // Aborts since output cannot be assembled without y.
        }
    }

    MPI_Gatherv(                                                                  // Starts a variable-count gather to collect uneven y segments into rank 0.
        ylocal, local_rows, MPI_DOUBLE,                                           // Sends this rank’s y segment sized to its owned row count.
        y, recvcountsY, displsY, MPI_DOUBLE,                                      // Receives into y on rank 0 using per-rank counts and displacements.
        0, MPI_COMM_WORLD                                                         // Sets rank 0 as the root and MPI_COMM_WORLD as the communicator.
    );

    if (rank == 0) {                                                              // Ensures only rank 0 performs output since only it owns the gathered y buffer.
        write_result("Result.txt", y, n);                                         // Writes the result vector in the specified space-separated format.
    }

    /* Cleanup */
    free(x);                                                                      // Releases the per-rank vector buffer after all computation and communication completes.
    free(Alocal);                                                                 // Releases the per-rank matrix slice buffer after local computation and no further access.
    free(ylocal);                                                                 // Releases the per-rank result slice buffer after it has been gathered.

    if (rank == 0) {                                                              // Restricts deallocation of root-only buffers to the rank that allocated them.
        free(Afull);                                                              // Releases the full matrix buffer now that scattering and computation are complete.
        free(y);                                                                  // Releases the gathered result buffer after writing to disk.
        free(sendcountsA);                                                        // Releases Scatterv send-count metadata allocated on rank 0.
        free(displsA);                                                            // Releases Scatterv displacement metadata allocated on rank 0.
        free(recvcountsY);                                                        // Releases Gatherv receive-count metadata allocated on rank 0.
        free(displsY);                                                            // Releases Gatherv displacement metadata allocated on rank 0.
    }

    MPI_Finalize();                                                               // Finalizes MPI to cleanly shut down the runtime and release MPI resources.
    return 0;                                                                     // Returns success to the OS since computation and output completed without fatal errors.
}
