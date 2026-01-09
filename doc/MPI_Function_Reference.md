# MPI Function Reference for `MPI_Bcast_Struct` Example

## Purpose
This document provides an authoritative reference for all MPI routines used in the `MPI_Bcast_Struct` example program.
Each function is linked directly to the **official MPI standard documentation** hosted by the MPI Forum and reputable MPI reference sites.
The intent is to support correctness, traceability, and academic rigor when studying or citing MPI-based implementations.

---

## MPI Initialization and Finalization

### `MPI_Init`
Initializes the MPI execution environment. This function must be called before any other MPI routine (except a small subset of inquiry functions).

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Init.html

---

### `MPI_Finalize`
Terminates the MPI execution environment. After this call, no further MPI communication is allowed.

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Finalize.html

---

## Communicator Inquiry Functions

### `MPI_Comm_size`
Determines the number of processes in a communicator.

**Used for:** determining the total number of MPI ranks participating in `MPI_COMM_WORLD`.

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Comm_size.html

---

### `MPI_Comm_rank`
Determines the rank (unique identifier) of the calling process within a communicator.

**Used for:** identifying the root process and controlling rank-specific behavior.

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Comm_rank.html

---

## Derived Datatype Construction

### `MPI_Type_create_struct`
Creates a user-defined MPI datatype describing a C struct composed of heterogeneous fields.

**Key characteristics:**
- Supports non-contiguous memory layouts
- Explicitly handles compiler-inserted padding
- Requires explicit block lengths, displacements, and base types

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Type_create_struct.html

---

### `MPI_Type_commit`
Commits a previously defined MPI datatype, making it usable in communication routines.

**Important note:**
All derived datatypes must be committed before use in point-to-point or collective operations.

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Type_commit.html

---

### `MPI_Type_free`
Frees an MPI datatype that is no longer needed.

**Used for:** proper resource management and avoiding datatype leaks in long-running MPI applications.

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Type_free.html

---

## Collective Communication

### `MPI_Bcast`
Broadcasts a message from one process (the root) to all other processes in a communicator.

**Usage in this example:**
- Broadcasts a single instance of a derived datatype (`SData`)
- Demonstrates correct collective usage with user-defined datatypes

**Collective semantics:**
- Must be called by **all ranks** in the communicator
- Arguments must be consistent across all ranks

**Official documentation:**
- https://www.mpi-forum.org/docs/
- https://www.mpich.org/static/docs/latest/www3/MPI_Bcast.html

---

## MPI Datatype and Addressing Types

### `MPI_Datatype`
Opaque handle representing an MPI datatype (predefined or user-defined).

**Official documentation:**
- https://www.mpi-forum.org/docs/

---

### `MPI_Aint`
Integer type capable of holding address displacements in a portable manner.

**Used for:** representing byte offsets computed with `offsetof`.

**Official documentation:**
- https://www.mpi-forum.org/docs/

---

## Related C Standard Facilities (Non-MPI)

### `offsetof`
Computes the byte offset of a structure member relative to the start of the structure.

**Relevance to MPI:**
- Essential for constructing correct derived datatypes
- Ensures portability across compilers and ABIs

**C standard reference:**
- https://en.cppreference.com/w/c/types/offsetof

---

## Summary Table

| Category                     | Function / Type              | Purpose |
|-----------------------------|------------------------------|---------|
| Initialization              | `MPI_Init`                   | Start MPI runtime |
| Finalization                | `MPI_Finalize`               | End MPI runtime |
| Communicator inquiry        | `MPI_Comm_size`              | Number of ranks |
| Communicator inquiry        | `MPI_Comm_rank`              | Rank identification |
| Derived datatype creation   | `MPI_Type_create_struct`     | Describe struct layout |
| Derived datatype management | `MPI_Type_commit`            | Activate datatype |
| Derived datatype management | `MPI_Type_free`              | Release datatype |
| Collective communication    | `MPI_Bcast`                  | Broadcast data |
| Addressing support          | `MPI_Aint`                   | Portable displacements |

---

## Concluding Remarks
This reference list reflects **best practices** for MPI programming involving composite data structures.
All linked functions are defined by the MPI standard and are implemented consistently across conforming MPI libraries such as **MPICH**, **Open MPI**, and **Microsoft MPI**.

This document is suitable for inclusion as:
- a project reference appendix,
- a teaching aid,
- or a formally cited artifact in academic work involving MPI.
