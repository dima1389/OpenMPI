/*
 * =====================================================================================
 * File:         <filename>.c
 * 
 * Purpose:      <Brief one-line description of what this program does>
 * 
 * Description:  <Detailed explanation of the program's purpose, algorithm, and approach.
 *                Include information about the parallelization strategy, data distribution,
 *                and computational workflow.>
 * 
 * MPI Features: <List the key MPI functions and concepts demonstrated>
 *               - MPI_Init / MPI_Finalize
 *               - MPI_Comm_rank / MPI_Comm_size
 *               - <Additional MPI functions used, e.g.:>
 *                 * MPI_Send / MPI_Recv
 *                 * MPI_Bcast
 *                 * MPI_Scatter / MPI_Gather
 *                 * MPI_Reduce
 *                 * MPI_Barrier
 *                 * MPI_Wtime
 *                 * MPI_Type_create_struct
 *                 * MPI_Alltoall
 *                 * MPI_Scatterv / MPI_Gatherv
 * 
 * Algorithm:    <High-level description of the algorithm or computational pattern>
 *               Example:
 *               1) Root process initializes data
 *               2) Data is distributed to all processes using MPI_Scatter
 *               3) Each process performs local computation
 *               4) Results are gathered back to root using MPI_Gather
 *               5) Root process displays or saves final results
 * 
 * Input:        <Describe input requirements>
 *               - No user input required (data generated internally), OR
 *               - Rank 0 prompts for: <description of expected input>
 *               - Command-line arguments: <description if applicable>
 *               - Input files: <filename and format description if applicable>
 * 
 * Output:       <Describe what the program outputs>
 *               - Console output: <description>
 *               - Output files: <filename and format if applicable>
 *               - Expected result format: <example or description>
 * 
 * Compilation:  <Platform-specific compilation commands>
 *               
 *               Linux (OpenMPI or MPICH):
 *                 mpicc -O2 -Wall <filename>.c -o <executable_name>
 *               
 *               Windows (Microsoft MPI with MinGW):
 *                 gcc <filename>.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o <executable_name>.exe
 *               
 *               Notes:
 *               - Use MPI compiler wrappers (mpicc) on Linux for automatic linking
 *               - On Windows, MSMPI_INC and MSMPI_LIB64 environment variables must be set
 *               - Additional flags: <any special compilation flags if needed>
 * 
 * Execution:    <How to run the program>
 *               
 *               Basic usage:
 *                 mpiexec -n <num_processes> <executable_name> [arguments]
 *               
 *               Examples:
 *                 mpiexec -n 4 <executable_name>
 *                 mpiexec -n 8 <executable_name> <input_file> <output_file>
 *               
 *               Requirements:
 *               - Minimum number of processes: <N or "1">
 *               - Maximum number of processes: <N or "unlimited">
 *               - Special constraints: <e.g., "Number of processes must divide N evenly">
 * 
 * Parameters:   <If program accepts command-line arguments, describe each>
 *               argv[1]: <description>
 *               argv[2]: <description>
 *               (Or state "None" if no arguments are required)
 * 
 * Dependencies: <Required libraries and headers>
 *               - <stdio.h>   : Standard I/O (printf, scanf, fprintf, etc.)
 *               - <stdlib.h>  : Memory allocation (malloc, free), rand, etc.
 *               - <string.h>  : String operations (strlen, strcpy, etc.)
 *               - <mpi.h>     : MPI library functions and types
 *               - <time.h>    : Time functions (time, clock, etc.)
 *               - <Additional headers if needed>
 * 
 * Notes:        <Important implementation details, assumptions, or limitations>
 *               - <Memory requirements or allocation strategy>
 *               - <Assumptions about data sizes or process counts>
 *               - <Known limitations or edge cases>
 *               - <Platform-specific behavior if applicable>
 * 
 * Safety:       <Important correctness and safety considerations>
 *               - Buffer sizes: <description of buffer management>
 *               - Error handling: <how errors are handled>
 *               - Synchronization: <any synchronization requirements>
 *               - Data validation: <input validation approach if applicable>
 * 
 * Example:      <Sample execution with expected output>
 *               
 *               $ mpiexec -n 4 <executable_name>
 *               <Expected output here>
 *               
 *               Explanation:
 *               <Brief explanation of the example output if needed>
 * 
 * Author:       <Your name or organization>
 * Created:      <Date>
 * Modified:     <Last modification date>
 * Version:      <Version number>
 * 
 * License:      <License information if applicable>
 * 
 * References:   <Citations, documentation links, or algorithm sources>
 *               - MPI Standard: https://www.mpi-forum.org/docs/
 *               - <Additional references>
 * 
 * =====================================================================================
 */
