/*
 * File:        mpi_<topic>.c
 *
 * Purpose:
 *   <One-line statement of what this example demonstrates.>
 *
 * Description:
 *   <Short paragraph describing the educational intent and what is observable/measurable.>
 *   This example focuses on <concept> and illustrates <behavior> under <conditions>.
 *
 * Key concepts:
 *   - <MPI concept: ranks, communicators, collectives, point-to-point, datatypes>
 *   - <Data distribution / ownership model>
 *   - <Synchronization / ordering / deadlock-avoidance aspect>
 *   - <Performance consideration: latency, bandwidth, imbalance, max-rank timing>
 *
 * Algorithm / workflow (high level):
 *   1) Rank 0 <initializes/reads input> and broadcasts/distributes it
 *   2) Each rank performs local computation on its assigned data
 *   3) Ranks exchange/collect results using <MPI primitive>
 *   4) Rank 0 reports results and/or timing
 *
 * OpenMP features used (list only those actually used in this file):
 *   - None (MPI-only example)
 *
 * MPI features used (list only those actually used in this file):
 *   - <e.g., MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size>
 *   - <e.g., MPI_Bcast / MPI_Scatter / MPI_Gather / MPI_Reduce / MPI_Barrier / MPI_Wtime>
 *   - <e.g., MPI_Send / MPI_Recv or MPI_Isend / MPI_Irecv / MPI_Waitall>
 *
 * Compilation:
 *   Linux (Open MPI / MPICH):
 *     mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_<topic>.c -o mpi_<topic>
 *   Windows (MS-MPI + MinGW, environment-dependent):
 *     gcc -O2 -Wall -Wextra -Wpedantic -g mpi_<topic>.c ^
 *         -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o mpi_<topic>.exe
 *
 * Execution:
 *   mpiexec -n <P> <executable> [args]
 *   Examples:
 *     mpiexec -n 4 ./mpi_<topic>
 *     mpiexec -n 8 ./mpi_<topic> [args]
 *
 * Inputs:
 *   - Command-line arguments: <describe argv usage or "None">
 *   - Optional interactive input: <rank 0 prompts for ...> (if applicable)
 *
 * References:
 *   - MPI Standard (MPI Forum): <relevant chapter/section>
 *   - Open MPI / MPICH documentation (mpiexec, runtime semantics)
 */
