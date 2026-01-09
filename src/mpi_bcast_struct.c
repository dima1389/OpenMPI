#include <stdio.h>    // printf, scanf
#include <mpi.h>      // MPI API
#include <stddef.h>   // offsetof

/*
 * Struct we want to send/receive as a single logical MPI object.
 *
 * IMPORTANT: In C, a struct may contain padding bytes inserted by the compiler
 * for alignment. Therefore, you should NOT assume fields are packed tightly.
 * Using offsetof() + MPI_Type_create_struct() makes the MPI datatype match
 * the exact in-memory layout produced by the compiler.
 */
typedef struct SData
{
    int    i1;  // integer field
    double d1;  // first double field
    double d2;  // second double field
} SData;

int main(int argc, char *argv[])
{
    int csize;   // communicator size (number of MPI processes)
    int prank;   // process rank (ID in [0..csize-1])

    /* Initialize MPI runtime. Must be called before most MPI functions. */
    MPI_Init(&argc, &argv);

    /* Query global communicator properties (MPI_COMM_WORLD). */
    MPI_Comm_size(MPI_COMM_WORLD, &csize);
    MPI_Comm_rank(MPI_COMM_WORLD, &prank);

    /*
     * Create an MPI derived datatype that describes SData.
     *
     * We will describe the struct as 3 "blocks":
     *   block 0: 1 x int
     *   block 1: 1 x double
     *   block 2: 1 x double
     *
     * Each block is defined by:
     *   - block length (how many items of that MPI type)
     *   - byte offset (where it begins inside the struct)
     *   - MPI type (MPI_INT, MPI_DOUBLE, ...)
     */
    MPI_Datatype data_t;                 // handle for the derived MPI datatype

    int lengths[3] = { 1, 1, 1 };        // number of items in each block

    MPI_Aint offsets[3];                 // displacements (byte offsets) for each block

    MPI_Datatype types[3] = {            // MPI type for each block
        MPI_INT,
        MPI_DOUBLE,
        MPI_DOUBLE
    };

    /*
     * Use offsetof(type, member) to compute member offsets safely.
     * This is the correct way to handle alignment/padding across compilers.
     */
    offsets[0] = (MPI_Aint)offsetof(SData, i1);
    offsets[1] = (MPI_Aint)offsetof(SData, d1);
    offsets[2] = (MPI_Aint)offsetof(SData, d2);

    /*
     * Build the struct datatype:
     *   count   = 3 blocks
     *   lengths = {1,1,1}
     *   offsets = {offset(i1), offset(d1), offset(d2)}
     *   types   = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE}
     *   data_t  = resulting datatype
     */
    MPI_Type_create_struct(3, lengths, offsets, types, &data_t);

    /*
     * Commit the datatype before use in communication.
     * After commit, MPI is allowed to optimize internal representations.
     */
    MPI_Type_commit(&data_t);

    SData s;  // instance to broadcast

    if (prank == 0)
    {
        /*
         * Root process reads the struct values from stdin.
         * Expected input format:
         *   <int> <double> <double>
         */
        scanf("%d %lf %lf", &s.i1, &s.d1, &s.d2);
    }

    /*
     * Broadcast the struct to all processes.
     * - buffer   = &s
     * - count    = 1 object
     * - datatype = data_t (our derived type describing SData layout)
     * - root     = 0
     * - comm     = MPI_COMM_WORLD
     *
     * After this call:
     *   - rank 0 has the original values
     *   - all other ranks have received identical values into their local 's'
     */
    MPI_Bcast(&s, 1, data_t, 0, MPI_COMM_WORLD);

    /* Each process prints the received struct. */
    printf("Process %d - Data %d %lf %lf\n", prank, s.i1, s.d1, s.d2);

    /*
     * Free the derived datatype once you no longer need it.
     * (Good hygiene; in long-running codes this matters.)
     */
    MPI_Type_free(&data_t);

    /* Finalize MPI runtime. No MPI calls after this (except a few allowed ones). */
    MPI_Finalize();

    return 0;
}
