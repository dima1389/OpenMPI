/*
 * File:
 *   mpi_<topic>.c
 *
 * Purpose:
 *   Demonstrate <controlled-term MPI concept>.
 *
 * Description:
 *   Mechanism:
 *     - <MPI mechanism / pattern used>
 *   Scenario / condition:
 *     - <what is assumed about ranks, root, inputs, topology, etc.>
 *   Observable outcome:
 *     - <what the user can see or measure to confirm behavior>
 *
 * Key concepts:
 *   - Communication: <ranks | communicators | collectives | point-to-point>
 *   - Semantics: <blocking | non-blocking | deterministic | nondeterministic>
 *   - Performance: <latency | bandwidth | load imbalance | synchronization cost | n/a>
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI environment
 *   2) <input acquisition / initialization>
 *   3) <communication pattern>
 *   4) <local computation>
 *   5) <collect / reduce / print results>
 *   6) Finalize MPI environment
 *
 * Correctness / Preconditions:
 *   - <P constraints: e.g., P >= 2>
 *   - <N constraints: e.g., N >= 1, N % P == 0 (if required)>
 *   - <collective participation: all ranks must call X>
 *
 * Output semantics:
 *   - <rank 0 only | all ranks | selected ranks>
 *   - Ordering: <deterministic | nondeterministic due to concurrent stdout>
 *
 * MPI features used (list only those actually used in this file):
 *   - <MPI_Init | MPI_Finalize | MPI_Comm_rank | MPI_Comm_size>
 *   - <MPI_Send | MPI_Recv | MPI_Isend | MPI_Irecv | MPI_Wait | MPI_Test>
 *   - <MPI_Bcast | MPI_Scatter | MPI_Gather | MPI_Reduce | MPI_Allreduce | MPI_Barrier | MPI_Wtime>
 *
 * Build / compile:
 *   Linux (Open MPI / MPICH):
 *     mpicc -O2 -Wall -Wextra -Wpedantic -g mpi_<topic>.c -o mpi_<topic>
 *
 *   Windows (MS-MPI SDK + MinGW/UCRT64; wrapper may vary by setup):
 *     gcc -O2 -Wall -Wextra -Wpedantic -g mpi_<topic>.c ^
 *         -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o mpi_<topic>.exe
 *
 * Run:
 *   mpiexec -n <P> <executable> [args]
 *
 * Inputs:
 *   - Command-line arguments: <...>
 *   - Interactive input: <rank 0 only | none>
 *
 * References:
 *   - MPI Standard: <collective / point-to-point / timing section>
 *   - Vendor docs (optional): <Open MPI / MPICH / MS-MPI notes if relevant>
 */
