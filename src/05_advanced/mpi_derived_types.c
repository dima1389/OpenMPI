/*
 * File:
 *   mpi_derived_types.c
 *
 * Purpose:
 *   Demonstrate creation and use of MPI derived datatypes for structured data.
 *
 * Description:
 *   This program defines a C struct containing mixed data types and constructs
 *   a corresponding MPI derived datatype using MPI_Type_create_struct().
 *
 *   Rank 0 initializes an instance of the struct and sends it to rank 1 using
 *   MPI_Send. Rank 1 receives the struct using MPI_Recv and prints its contents.
 *
 *   The example illustrates why derived datatypes are required in C:
 *     - C structs may contain padding inserted by the compiler
 *     - MPI must be informed of the exact in-memory layout
 *
 * Key concepts:
 *   - MPI derived datatypes
 *   - Memory layout and padding
 *   - MPI_Type_create_struct
 *   - MPI_Type_commit / MPI_Type_free
 *
 * Algorithm / workflow (high level):
 *   1) Define a C struct with heterogeneous fields
 *   2) Initialize MPI runtime
 *   3) Query rank and communicator size
 *   4) Create and commit an MPI derived datatype matching the struct layout
 *   5) Rank 0 sends the struct to rank 1
 *   6) Rank 1 receives and prints the struct
 *   7) Free the derived datatype
 *   8) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Type_create_struct
 *   - MPI_Type_commit
 *   - MPI_Type_free
 *   - MPI_Send
 *   - MPI_Recv
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_derived_types.c -o mpi_derived_types
 *
 * Run:
 *   mpirun -n 2 ./mpi_derived_types
 */

#include <stdio.h>    /* Provides printf for formatted output */
#include <stddef.h>   /* Provides offsetof for portable struct layout calculation */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * DataRecord
 *
 * Heterogeneous data structure used to demonstrate MPI derived datatypes.
 * The compiler may insert padding between fields to satisfy alignment rules.
 */
typedef struct DataRecord
{
    int    id;        /* Integer identifier field */
    double value;     /* Floating-point value field */
    double weight;   /* Additional floating-point field */
} DataRecord;

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                /* Stores the rank (ID) of the calling process */
    int size;                /* Stores the total number of processes */
    DataRecord record;       /* Instance of the structured data */
    MPI_Datatype mpi_record; /* MPI derived datatype describing DataRecord */

    /* Initialize the MPI execution environment to enable datatype creation and communication. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Enforce the requirement that this example uses exactly two ranks. */
    if (size < 2) {
        if (rank == 0) {
            printf("ERROR: mpi_derived_types requires at least 2 processes\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1); /* Abort all ranks consistently on invalid configuration. */
    }

    /* Define the number of fields in the struct for datatype construction. */
    {
        const int nitems = 3; /* Specifies the number of struct members. */

        int blocklengths[3]; /* Describes the number of elements in each field. */
        MPI_Datatype types[3]; /* Describes the MPI datatype of each field. */
        MPI_Aint offsets[3]; /* Describes the byte offset of each field in the struct. */

        /* Define block lengths corresponding to scalar struct members. */
        blocklengths[0] = 1; /* One integer element for id. */
        blocklengths[1] = 1; /* One double element for value. */
        blocklengths[2] = 1; /* One double element for weight. */

        /* Define MPI datatypes matching the C field types. */
        types[0] = MPI_INT;    /* Maps C int to MPI_INT. */
        types[1] = MPI_DOUBLE; /* Maps C double to MPI_DOUBLE. */
        types[2] = MPI_DOUBLE; /* Maps C double to MPI_DOUBLE. */

        /* Compute field offsets using offsetof to respect compiler-inserted padding. */
        offsets[0] = offsetof(DataRecord, id);     /* Offset of id field. */
        offsets[1] = offsetof(DataRecord, value);  /* Offset of value field. */
        offsets[2] = offsetof(DataRecord, weight); /* Offset of weight field. */

        /* Create the MPI derived datatype that matches the in-memory layout of DataRecord. */
        MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_record);

        /* Commit the derived datatype so it can be used in communication calls. */
        MPI_Type_commit(&mpi_record);
    }

    /* Execute the sender role on rank 0. */
    if (rank == 0) {
        /* Initialize the struct fields with deterministic values. */
        record.id = 1;          /* Assigns a unique identifier. */
        record.value = 3.1415;  /* Assigns a representative floating-point value. */
        record.weight = 2.7183; /* Assigns a second floating-point value. */

        /* Send the structured data to rank 1 using the derived datatype. */
        MPI_Send(&record, 1, mpi_record, 1, 0, MPI_COMM_WORLD);

        /* Report the transmitted values for traceability. */
        printf("Rank 0 sent record: id=%d, value=%f, weight=%f\n",
               record.id, record.value, record.weight);
    }

    /* Execute the receiver role on rank 1. */
    if (rank == 1) {
        /* Receive the structured data from rank 0 using the derived datatype. */
        MPI_Recv(&record, 1, mpi_record, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Print the received struct fields to verify correctness. */
        printf("Rank 1 received record: id=%d, value=%f, weight=%f\n",
               record.id, record.value, record.weight);
    }

    /* Free the derived datatype once it is no longer needed. */
    MPI_Type_free(&mpi_record);

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
