/*
 * File:
 *   mpi_ping_pong.c
 *
 * Purpose:
 *   Demonstrate round-trip point-to-point communication (ping-pong) and estimate message latency.
 *
 * Description:
 *   This program performs a classic MPI "ping-pong" benchmark between rank 0 and rank 1:
 *     - Rank 0 sends a message ("ping") to rank 1.
 *     - Rank 1 immediately sends the message back ("pong") to rank 0.
 *     - Rank 0 measures the round-trip time (RTT) using MPI_Wtime().
 *
 *   The exchange is repeated for a configurable number of iterations, and rank 0 reports:
 *     - average RTT
 *     - estimated one-way latency as RTT / 2
 *
 * Key concepts:
 *   - Blocking point-to-point communication
 *   - Round-trip latency measurement
 *   - Synchronization effects (warm-up + barrier)
 *   - Avoiding timing noise by measuring many iterations
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Validate that at least 2 ranks exist
 *   4) Rank 0 configures message size and iteration count (argv or defaults)
 *   5) Broadcast parameters to all ranks
 *   6) Allocate message buffers
 *   7) Warm-up ping-pong exchanges to reduce first-iteration effects
 *   8) Barrier synchronize before timing
 *   9) Timed ping-pong loop for a fixed number of iterations
 *  10) Rank 0 prints RTT and estimated one-way latency
 *  11) Free resources and finalize MPI
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Barrier
 *   - MPI_Wtime
 *   - MPI_Send
 *   - MPI_Recv
 *   - MPI_Abort
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_ping_pong.c -o mpi_ping_pong
 *
 * Run:
 *   mpirun -n 2 ./mpi_ping_pong [bytes] [iterations]
 *
 * Parameters:
 *   bytes       - message payload size in bytes (default: 8)
 *   iterations  - number of timed ping-pong iterations (default: 10000)
 *
 * Notes:
 *   - For clarity and portability, MPI_BYTE is used for the payload datatype.
 *   - The program prints results only on rank 0 to avoid output interleaving.
 */

#include <stdio.h>    /* Provides printf and fprintf for output */
#include <stdlib.h>   /* Provides malloc, free, and strtol */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;                    /* Stores the rank (ID) of the calling process */
    int size;                    /* Stores the total number of processes */
    int msg_bytes = 0;           /* Stores message payload size in bytes */
    int iterations = 0;          /* Stores number of timed iterations */
    int warmup = 0;              /* Stores number of warm-up iterations */
    unsigned char *buf = NULL;   /* Stores the message payload buffer */
    const int peer0 = 0;         /* Defines the first participant rank */
    const int peer1 = 1;         /* Defines the second participant rank */
    const int tag = 0;           /* Defines a message tag used for matching send/receive */
    double t0 = 0.0;             /* Stores timing start timestamp */
    double t1 = 0.0;             /* Stores timing end timestamp */
    double total = 0.0;          /* Stores total elapsed time across timed iterations */
    double avg_rtt = 0.0;        /* Stores average round-trip time (seconds) */
    double avg_one_way = 0.0;    /* Stores estimated one-way latency (seconds) */

    /* Initialize the MPI execution environment to enable communication and timing. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Enforce the requirement that at least ranks 0 and 1 exist for a two-party exchange. */
    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: mpi_ping_pong requires at least 2 processes (got %d)\n", size); /* Reports configuration error. */
        }
        MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks consistently on invalid configuration. */
    }

    /* Configure benchmark parameters on rank 0 to keep argument parsing centralized. */
    if (rank == 0) {
        msg_bytes = 8;        /* Sets a small default payload size suitable for latency measurement. */
        iterations = 10000;   /* Sets a default iteration count to reduce timing noise. */

        /* Parse optional message size argument to allow experimentation with different payloads. */
        if (argc >= 2) {
            char *end = NULL;                 /* Holds parsing end pointer for validation. */
            long tmp = strtol(argv[1], &end, 10); /* Converts argv[1] to an integer byte count. */

            /* Validate that parsing succeeded and produced a positive byte count. */
            if (end == argv[1] || *end != '\0' || tmp <= 0) {
                fprintf(stderr, "Usage: %s [bytes] [iterations]\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks consistently on invalid input. */
            }

            msg_bytes = (int)tmp; /* Stores the validated message size in bytes. */
        }

        /* Parse optional iteration count argument to allow controlling measurement duration. */
        if (argc >= 3) {
            char *end = NULL;                 /* Holds parsing end pointer for validation. */
            long tmp = strtol(argv[2], &end, 10); /* Converts argv[2] to an iteration count. */

            /* Validate that parsing succeeded and produced a positive iteration count. */
            if (end == argv[2] || *end != '\0' || tmp <= 0) {
                fprintf(stderr, "Usage: %s [bytes] [iterations]\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks consistently on invalid input. */
            }

            iterations = (int)tmp; /* Stores the validated number of timed iterations. */
        }
    }

    /* Broadcast benchmark parameters so both participating ranks allocate consistent buffers. */
    MPI_Bcast(&msg_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Derive a small warm-up count to reduce cold-start effects without dominating runtime. */
    warmup = (iterations < 100) ? iterations : 100; /* Bounds warm-up iterations to a small fixed maximum. */

    /* Allocate the payload buffer on each rank to match the configured message size. */
    buf = (unsigned char *)malloc((size_t)msg_bytes); /* Allocates a raw byte buffer for MPI_BYTE transfers. */

    /* Validate buffer allocation to avoid undefined behavior in send/receive calls. */
    if (buf == NULL) {
        fprintf(stderr, "Rank %d: ERROR: failed to allocate %d-byte buffer\n", rank, msg_bytes); /* Reports allocation failure. */
        MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks because both must participate in communication. */
    }

    /* Initialize buffer contents deterministically to avoid reading uninitialized memory. */
    for (int i = 0; i < msg_bytes; ++i) {
        buf[i] = (unsigned char)(i & 0xFF); /* Fills each byte with a simple pattern for defined payload content. */
    }

    /* Execute warm-up ping-pong exchanges to stabilize runtime behavior before measurement. */
    for (int it = 0; it < warmup; ++it) {
        if (rank == peer0) {
            MPI_Send(buf, msg_bytes, MPI_BYTE, peer1, tag, MPI_COMM_WORLD); /* Sends the payload to rank 1. */
            MPI_Recv(buf, msg_bytes, MPI_BYTE, peer1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* Receives the payload back from rank 1. */
        } else if (rank == peer1) {
            MPI_Recv(buf, msg_bytes, MPI_BYTE, peer0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* Receives the payload from rank 0. */
            MPI_Send(buf, msg_bytes, MPI_BYTE, peer0, tag, MPI_COMM_WORLD); /* Sends the payload back to rank 0. */
        }
    }

    /* Synchronize all ranks so the timed section starts from a comparable program point. */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Start timing on rank 0 immediately before the timed ping-pong loop. */
    if (rank == peer0) {
        t0 = MPI_Wtime(); /* Captures the start timestamp for the RTT measurement interval. */
    }

    /* Execute the timed ping-pong loop for the configured number of iterations. */
    for (int it = 0; it < iterations; ++it) {
        if (rank == peer0) {
            MPI_Send(buf, msg_bytes, MPI_BYTE, peer1, tag, MPI_COMM_WORLD); /* Sends a ping message to rank 1. */
            MPI_Recv(buf, msg_bytes, MPI_BYTE, peer1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* Receives the pong response from rank 1. */
        } else if (rank == peer1) {
            MPI_Recv(buf, msg_bytes, MPI_BYTE, peer0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* Receives a ping message from rank 0. */
            MPI_Send(buf, msg_bytes, MPI_BYTE, peer0, tag, MPI_COMM_WORLD); /* Sends the pong response back to rank 0. */
        }
    }

    /* Stop timing on rank 0 immediately after the timed loop completes. */
    if (rank == peer0) {
        t1 = MPI_Wtime(); /* Captures the end timestamp for the RTT measurement interval. */
        total = t1 - t0;  /* Computes the total time for all ping-pong iterations. */
    }

    /* Compute and print results on rank 0 to avoid non-deterministic interleaved output. */
    if (rank == peer0) {
        avg_rtt = total / (double)iterations;        /* Computes the mean round-trip time per iteration. */
        avg_one_way = avg_rtt / 2.0;                 /* Estimates one-way latency assuming symmetric path and processing. */

        printf("MPI ping-pong (rank 0 <-> rank 1)\n"); /* Labels the benchmark context. */
        printf("Message size: %d bytes\n", msg_bytes); /* Reports the payload size used. */
        printf("Iterations:   %d\n", iterations);      /* Reports the number of timed iterations. */
        printf("Avg RTT:      %.9f s\n", avg_rtt);     /* Reports average round-trip time per iteration. */
        printf("Avg one-way:  %.9f s (RTT/2)\n", avg_one_way); /* Reports estimated one-way latency. */
    }

    /* Release the payload buffer to avoid memory leaks. */
    free(buf); /* Frees the allocated message buffer. */

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
