#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

/*
 * MPI Vector Multiplication
 *
 * Each process computes a block of:
 *   C[i] = A[i] * B[i]
 *
 * Demonstrates:
 *  - Random data generation on root
 *  - Block-wise data distribution (MPI_Scatter)
 *  - Local computation
 *  - Result collection (MPI_Gather)
 */

int main(int argc, char *argv[])
{
    int rank, size;
    int N = 16;               /* Total vector length (must be divisible by size) */
    int local_n;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    double *local_A;
    double *local_B;
    double *local_C;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Ensure N is divisible by number of processes */
    if (N % size != 0) {
        if (rank == 0)
            printf("Vector size N must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    local_n = N / size;

    /* Allocate local buffers */
    local_A = (double *)malloc(local_n * sizeof(double));
    local_B = (double *)malloc(local_n * sizeof(double));
    local_C = (double *)malloc(local_n * sizeof(double));

    /* Root process allocates and initializes full vectors */
    if (rank == 0) {
        A = (double *)malloc(N * sizeof(double));
        B = (double *)malloc(N * sizeof(double));
        C = (double *)malloc(N * sizeof(double));

        srand((unsigned int)time(NULL));
        for (int i = 0; i < N; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    }

    /* Distribute vector segments */
    MPI_Scatter(A, local_n, MPI_DOUBLE,
                local_A, local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, local_n, MPI_DOUBLE,
                local_B, local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Local vector multiplication */
    for (int i = 0; i < local_n; i++) {
        local_C[i] = local_A[i] * local_B[i];
    }

    /* Gather results */
    MPI_Gather(local_C, local_n, MPI_DOUBLE,
               C, local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    /* Print results on root */
    if (rank == 0) {
        printf("Vector A:\n");
        for (int i = 0; i < N; i++)
            printf("%5.1f ", A[i]);
        printf("\n\n");

        printf("Vector B:\n");
        for (int i = 0; i < N; i++)
            printf("%5.1f ", B[i]);
        printf("\n\n");

        printf("Vector C = A * B:\n");
        for (int i = 0; i < N; i++)
            printf("%5.1f ", C[i]);
        printf("\n");

        free(A);
        free(B);
        free(C);
    }

    /* Cleanup */
    free(local_A);
    free(local_B);
    free(local_C);

    MPI_Finalize();
    return 0;
}
