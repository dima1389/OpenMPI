/*
 * File:
 *   mpi_<topic>.c
 *
 * Purpose:
 *   Demonstrate <controlled-term MPI concept>.
 *
 * Description:
 *   This example demonstrates <controlled-term MPI mechanism> under
 *   <controlled-term condition>.
 *   Observable outcome: <controlled-term behavior that can be observed or measured>.
 *
 * Key concepts:
 *   - <ranks | communicators | collectives | point-to-point>
 *   - <deterministic | nondeterministic | blocking | non-blocking behavior>
 *   - <performance effect: latency | bandwidth | imbalance>
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI environment and input data
 *   2) Distribute data among ranks
 *   3) Perform local computation
 *   4) Collect results and report output
 *
 * MPI features used (list only those actually used in this file):
 *   - <MPI_Init | MPI_Finalize | MPI_Comm_rank | MPI_Comm_size>
 *   - <MPI_Bcast | MPI_Scatter | MPI_Gather | MPI_Reduce | MPI_Barrier | MPI_Wtime>
 *
 * Compilation:
 *   mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_<topic>.c -o mpi_<topic>
 *
 * Execution:
 *   mpiexec -n <P> <executable> [arguments]
 *
 * Inputs:
 *   - Command-line arguments: <N | none>
 *   - Interactive input: <rank 0 only | none>
 *
 * References:
 *   - MPI Standard: <chapter or section>
 */
