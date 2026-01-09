/*
 * File:
 *   mpi_bcast_struct_derived_type.c
 *
 * Purpose:
 *   Demonstrate broadcasting a C struct by defining and using an MPI derived datatype.
 *
 * Description:
 *   This example demonstrates creating an MPI datatype that matches a compiler-defined C struct
 *   memory layout (including any padding) using offsetof() and MPI_Type_create_struct(), then
 *   broadcasting a single struct instance from rank 0 to all ranks using MPI_Bcast().
 *   Observable outcome: all ranks print identical struct field values after the broadcast.
 *
 * Key concepts:
 *   - ranks, communicators, collectives (broadcast)
 *   - derived datatypes, alignment/padding correctness, portable layout description
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI and query rank/size
 *   2) Define and commit an MPI derived datatype describing the struct layout
 *   3) Read struct values on rank 0
 *   4) Broadcast the struct to all ranks and print the received values
 *   5) Free the datatype and finalize MPI
 *
 * MPI features used (list only those actually used in this file):
 *   - MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size
 *   - MPI_Type_create_struct, MPI_Type_commit, MPI_Type_free
 *   - MPI_Bcast
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_bcast_struct_derived_type.c -o mpi_bcast_struct_derived_type
 *
 * Execution:
 *   mpiexec -n <P> mpi_bcast_struct_derived_type
 *
 * Inputs:
 *   - Command-line arguments: none
 *   - Interactive input: rank 0 only, format: <int> <double> <double>
 *
 * References:
 *   - MPI Standard: Derived datatypes (struct types) and collective communication (broadcast)
 */

#include <stdio.h>    // Provides printf(), scanf() for I/O used in rank 0 input and per-rank output.
#include <mpi.h>      // Provides MPI_Init(), MPI_Bcast(), MPI datatype creation APIs, and MPI types.
#include <stddef.h>   // Provides offsetof() for computing portable member displacements within a struct.

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
    int    i1;  // Stores the integer field that must be described as MPI_INT in the derived type.
    double d1;  // Stores the first floating-point field that must be described as MPI_DOUBLE.
    double d2;  // Stores the second floating-point field that must be described as MPI_DOUBLE.
} SData;

int main(int argc, char *argv[])
{
    int csize;   // Holds the number of processes in MPI_COMM_WORLD for rank bounds and logic.
    int prank;   // Holds the calling process rank in MPI_COMM_WORLD to branch on root/non-root.

    MPI_Init(&argc, &argv); // Initializes the MPI execution environment and enables MPI calls.

    MPI_Comm_size(MPI_COMM_WORLD, &csize); // Queries communicator size to determine total ranks.
    MPI_Comm_rank(MPI_COMM_WORLD, &prank); // Queries communicator rank to identify this process.

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
    MPI_Datatype data_t;                 // Declares a handle that will represent the derived datatype.

    int lengths[3] = { 1, 1, 1 };        // Specifies one element per block to match each struct field.

    MPI_Aint offsets[3];                 // Declares byte displacements for each member within SData.

    MPI_Datatype types[3] = {            // Specifies the MPI elemental types corresponding to each field.
        MPI_INT,
        MPI_DOUBLE,
        MPI_DOUBLE
    };

    offsets[0] = (MPI_Aint)offsetof(SData, i1); // Computes the byte offset of i1 to capture padding/alignment.
    offsets[1] = (MPI_Aint)offsetof(SData, d1); // Computes the byte offset of d1 to capture padding/alignment.
    offsets[2] = (MPI_Aint)offsetof(SData, d2); // Computes the byte offset of d2 to capture padding/alignment.

    MPI_Type_create_struct(3, lengths, offsets, types, &data_t); // Builds a struct datatype matching SData layout.
    MPI_Type_commit(&data_t);                                   // Commits the datatype so it can be used in MPI calls.

    SData s;  // Allocates the local struct instance that will be populated (root) and received (all ranks).

    if (prank == 0)
    {
        scanf("%d %lf %lf", &s.i1, &s.d1, &s.d2); // Reads the struct fields on the root rank from standard input.
    }

    MPI_Bcast(&s, 1, data_t, 0, MPI_COMM_WORLD); // Broadcasts the struct instance from rank 0 to all ranks.

    printf("Process %d - Data %d %lf %lf\n", prank, s.i1, s.d1, s.d2); // Prints per-rank confirmation of received values.

    MPI_Type_free(&data_t); // Frees the derived datatype handle to release MPI internal resources.

    MPI_Finalize(); // Finalizes the MPI environment and invalidates most subsequent MPI calls.

    return 0; // Returns success status to the host environment.
}
