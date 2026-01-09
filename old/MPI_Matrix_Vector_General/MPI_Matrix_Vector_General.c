#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)
{
    if (rank == 0) {
        fprintf(stderr, "ERROR: %s\n", msg);
    }
    MPI_Abort(comm, 1);
}

/* Count how many doubles are present in a file (vector size). */
static int count_doubles_in_file(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (!f) return -1;

    int count = 0;
    double tmp;
    while (fscanf(f, "%lf", &tmp) == 1) {
        count++;
    }
    fclose(f);
    return count;
}

static double *load_vector(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) { fclose(f); return NULL; }

    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%lf", &x[i]) != 1) {
            free(x);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    return x;
}

static double *load_matrix(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    size_t m = (size_t)n * (size_t)n;
    double *A = (double *)malloc(m * sizeof(double));
    if (!A) { fclose(f); return NULL; }

    for (size_t i = 0; i < m; i++) {
        if (fscanf(f, "%lf", &A[i]) != 1) {
            free(A);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    return A;
}

static void write_result(const char *fname, const double *y, int n)
{
    FILE *f = fopen(fname, "w");
    if (!f) return;

    for (int i = 0; i < n; i++) {
        fprintf(f, "%lf%s", y[i], (i + 1 == n) ? "" : " ");
    }
    fprintf(f, "\n");
    fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <vector_file> <matrix_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *vec_file = argv[1];
    const char *mat_file = argv[2];

    int n = 0;

    /* Rank 0 determines n from the vector file. */
    if (rank == 0) {
        n = count_doubles_in_file(vec_file);
        if (n <= 0) {
            fprintf(stderr, "ERROR: cannot read vector size from file '%s'\n", vec_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Broadcast n to all ranks. */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute uneven row distribution: rows_i and offset_i for each rank i. */
    int q = n / p;
    int r = n % p;

    int local_rows = q + (rank < r ? 1 : 0);
    int local_row_offset = rank * q + (rank < r ? rank : r);

    /* Prepare counts/displacements for Scatterv/Gatherv (rank 0 needs full arrays). */
    int *sendcountsA = NULL;
    int *displsA     = NULL;
    int *recvcountsY = NULL;
    int *displsY     = NULL;

    if (rank == 0) {
        sendcountsA = (int *)malloc((size_t)p * sizeof(int));
        displsA     = (int *)malloc((size_t)p * sizeof(int));
        recvcountsY = (int *)malloc((size_t)p * sizeof(int));
        displsY     = (int *)malloc((size_t)p * sizeof(int));

        if (!sendcountsA || !displsA || !recvcountsY || !displsY) {
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for counts/displacements");
        }

        int dispA = 0;
        int dispY = 0;
        for (int i = 0; i < p; i++) {
            int rows_i = q + (i < r ? 1 : 0);

            sendcountsA[i] = rows_i * n;  /* matrix chunk is rows_i rows, each row has n cols */
            displsA[i]     = dispA;

            recvcountsY[i] = rows_i;      /* result chunk is rows_i entries */
            displsY[i]     = dispY;

            dispA += sendcountsA[i];
            dispY += recvcountsY[i];
        }
    }

    /* Allocate and load x (broadcast to all). */
    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) {
        die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for vector x");
    }

    if (rank == 0) {
        double *tmp = load_vector(vec_file, n);
        if (!tmp) {
            free(x);
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read vector file (format/size mismatch)");
        }
        for (int i = 0; i < n; i++) x[i] = tmp[i];
        free(tmp);
    }

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Rank 0 loads full matrix A; others keep NULL. */
    double *Afull = NULL;
    if (rank == 0) {
        Afull = load_matrix(mat_file, n);
        if (!Afull) {
            free(x);
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read matrix file (format/size mismatch)");
        }
    }

    /* Allocate local matrix chunk: local_rows * n */
    double *Alocal = NULL;
    if (local_rows > 0) {
        Alocal = (double *)malloc((size_t)local_rows * (size_t)n * sizeof(double));
        if (!Alocal) {
            free(x);
            if (rank == 0) free(Afull);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local matrix chunk");
        }
    }

    /* Scatter uneven row blocks of A. */
    MPI_Scatterv(
        Afull, sendcountsA, displsA, MPI_DOUBLE,
        Alocal, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    /* Compute local result y_local = A_local * x */
    double *ylocal = NULL;
    if (local_rows > 0) {
        ylocal = (double *)malloc((size_t)local_rows * sizeof(double));
        if (!ylocal) {
            free(x);
            if (rank == 0) free(Afull);
            free(Alocal);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local result chunk");
        }

        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            const double *row = &Alocal[(size_t)i * (size_t)n];
            for (int j = 0; j < n; j++) {
                sum += row[j] * x[j];
            }
            ylocal[i] = sum;
        }
    }

    /* Gather uneven y chunks to rank 0. */
    double *y = NULL;
    if (rank == 0) {
        y = (double *)malloc((size_t)n * sizeof(double));
        if (!y) {
            free(x);
            free(Afull);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for full result y");
        }
    }

    MPI_Gatherv(
        ylocal, local_rows, MPI_DOUBLE,
        y, recvcountsY, displsY, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        write_result("Result.txt", y, n);
    }

    /* Cleanup */
    free(x);
    free(Alocal);
    free(ylocal);

    if (rank == 0) {
        free(Afull);
        free(y);
        free(sendcountsA);
        free(displsA);
        free(recvcountsY);
        free(displsY);
    }

    MPI_Finalize();
    return 0;
}
