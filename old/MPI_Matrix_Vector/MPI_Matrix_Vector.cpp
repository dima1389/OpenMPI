#include <stdio.h>   // For FILE*, fopen, fscanf, fprintf, fclose
#include <mpi.h>     // MPI library

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
int returnSize(char* fname)
{
    FILE* f = fopen(fname, "r");
    int dim = 0;
    double tmp;

    // Read doubles one by one until EOF and count them
    while (fscanf(f, "%lf", &tmp) != EOF)
        dim++;

    fclose(f);
    return dim;
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
//   Caller is responsible for delete[].
// -----------------------------------------------------------------------------
double* loadVec(char* fname, int n)
{
    FILE* f = fopen(fname, "r");
    double* res = new double[n]; // allocate vector
    double* it = res;

    // Read values into consecutive elements of res
    while (fscanf(f, "%lf", it++) != EOF)
        /* empty body */;

    fclose(f);
    return res;
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
//   Caller is responsible for delete[].
// -----------------------------------------------------------------------------
double* loadMat(char* fname, int n)
{
    FILE* f = fopen(fname, "r");
    double* res = new double[n * n]; // allocate matrix as 1D array
    double* it = res;

    // Read n*n values into res
    while (fscanf(f, "%lf", it++) != EOF)
        /* empty body */;

    fclose(f);
    return res;
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
void logRes(const char* fname, double* res, int n)
{
    FILE* f = fopen(fname, "w");
    for (int i = 0; i != n; ++i)
        fprintf(f, "%lf ", res[i]);
    fclose(f);
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
//
// Steps:
//   1. Rank 0 reads the vector file to determine dim (vector length).
//   2. Broadcast dim to all ranks.
//   3. Rank 0 loads the full vector; broadcast it to all ranks.
//   4. Rank 0 loads the full matrix; scatter pieces of it to each rank.
//   5. Each rank computes its partial result (subset of rows).
//   6. Gather all partial results to rank 0.
//   7. Rank 0 writes the full result vector to "res.txt".
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int csize;  // total number of MPI processes
    int prank;  // rank (ID) of this MPI process

    // Initialize MPI and get communicator size and rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    // Command line arguments: vector file and matrix file
    char* vfname = argv[1];
    char* mfname = argv[2];

    int dim;        // dimension of the vector/matrix
    double* mat;    // local chunk of matrix
    double* vec;    // full vector (every process has a copy)
    double* tmat;   // full matrix (only rank 0 has it)
    double* lres;   // local result (subset of rows)
    double* res;    // final result (only rank 0 has it)

    // Rank 0 reads vector file to determine dimension
    if (prank == 0)
        dim = returnSize(vfname);

    // Broadcast the dimension to all processes
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Load or allocate vector:
    // Rank 0 reads full vector from file; others just allocate memory.
    if (prank == 0)
        vec = loadVec(vfname, dim);
    else
        vec = new double[dim];

    // Broadcast full vector to all ranks
    MPI_Bcast(vec, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 loads full matrix (dim x dim)
    if (prank == 0)
        tmat = loadMat(mfname, dim);

    // Number of matrix elements per process:
    // We are dividing the matrix by rows, so each process gets
    // (dim / csize) rows, each with dim columns.
    // Total elements per process = dim * (dim / csize)
    int msize = dim * dim / csize;
    mat = new double[msize];

    // Scatter parts of the matrix from rank 0 to all ranks:
    //   tmat (root buffer) -> mat (local buffer on each process)
    MPI_Scatter(
        tmat, msize, MPI_DOUBLE,   // send buffer (root only)
        mat,  msize, MPI_DOUBLE,   // receive buffer (all)
        0, MPI_COMM_WORLD
    );

    // Each process will compute "to" rows of the result
    int to = dim / csize;
    lres = new double[to];

    // Local matrix-vector multiplication:
    // Here 'mat' contains 'to' consecutive rows of the global matrix.
    // For each local row i, compute:
    //   lres[i] = sum_j mat[i * dim + j] * vec[j]
    for (int i = 0; i != to; ++i)
    {
        double s = 0;
        for (int j = 0; j != dim; ++j)
            s += mat[i * dim + j] * vec[j];
        lres[i] = s;
    }

    // Rank 0 allocates space for the complete result vector
    if (prank == 0)
        res = new double[dim];

    // Gather partial results from all processes into res on rank 0
    MPI_Gather(
        lres, to, MPI_DOUBLE,   // send buffer on each rank
        res,  to, MPI_DOUBLE,   // recv buffer on root
        0, MPI_COMM_WORLD
    );

    // Rank 0 logs the final result to a file
    if (prank == 0)
    {
        logRes("Result.txt", res, dim);
    }

    // Clean-up: free dynamically allocated memory
    if (prank == 0)
    {
        delete[] tmat;
        delete[] res;
    }

    delete[] vec;
    delete[] mat;
    delete[] lres;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
