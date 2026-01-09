/*
 * File:
 *   mpi_ring.c
 *
 * Purpose:
 *   Demonstrate ring communication and deadlock-safe point-to-point messaging.
 *
 * Description:
 *   This program implements a logical ring over MPI_COMM_WORLD ranks:
 *     left  = (rank - 1 + size) % size
 *     right = (rank + 1) % size
 *
 *   Each rank sends an integer token to its right neighbor and receives a token
 *   from its left neighbor. The token is initialized on rank 0 and circulates
 *   around the ring for a configurable number of hops.
 *
 *   To avoid deadlock with blocking calls, this implementation uses MPI_Sendrecv,
 *   which performs a send and receive as a single matched operation.
 *
 * Key concepts:
 *   - Logical process topologies (ring)
 *   - Point-to-point communication pattern design
 *   - Deadlock avoidance with MPI_Sendrecv
 *   - Token passing and hop counting
 *
 * Algorithm / workflow (high level):
 *   1) Initialize MPI runtime
 *   2) Query rank and communicator size
 *   3) Compute left/right neighbor ranks in the ring
 *   4) Configure number of hops on rank 0 and broadcast it
 *   5) Initialize token on rank 0 (others use a sentinel)
 *   6) Repeat for hop = 0..hops-1:
 *        - Send token to right and receive token from left (MPI_Sendrecv)
 *        - Optionally modify token to reflect traversal
 *   7) Rank 0 reports final token value
 *   8) Finalize MPI runtime
 *
 * MPI features used:
 *   - MPI_Init
 *   - MPI_Comm_rank
 *   - MPI_Comm_size
 *   - MPI_Bcast
 *   - MPI_Sendrecv
 *   - MPI_Finalize
 *
 * Build / compile:
 *   mpicc mpi_ring.c -o mpi_ring
 *
 * Run:
 *   mpirun -n <p> ./mpi_ring [hops]
 *
 * Parameters:
 *   hops - number of ring exchanges to perform (default: size)
 *
 * Notes:
 *   - If hops == size, the token makes one full traversal of the ring.
 *   - Output is restricted to rank 0 to avoid stdout interleaving.
 */

#include <stdio.h>    /* Provides printf and fprintf for output */
#include <stdlib.h>   /* Provides strtol for command-line parsing */
#include <mpi.h>      /* Provides MPI API declarations */

/*
 * main
 *
 * Entry point executed independently by every MPI process.
 */
int main(int argc, char *argv[])
{
    int rank;               /* Stores the rank (ID) of the calling process */
    int size;               /* Stores the total number of processes */
    int left;               /* Stores the rank of the left neighbor */
    int right;              /* Stores the rank of the right neighbor */
    int hops = 0;           /* Stores the number of token exchanges to perform */
    int token = 0;          /* Stores the token value carried around the ring */
    const int tag = 0;      /* Defines the message tag used for matching send/receive */

    /* Initialize the MPI execution environment to enable point-to-point communication. */
    MPI_Init(&argc, &argv);

    /* Determine this process rank within the global communicator. */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes participating in MPI_COMM_WORLD. */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute neighbor ranks in the logical ring topology. */
    left = (rank - 1 + size) % size;  /* Computes the source rank for incoming token messages. */
    right = (rank + 1) % size;        /* Computes the destination rank for outgoing token messages. */

    /* Configure the hop count on rank 0 to keep argument parsing centralized. */
    if (rank == 0) {
        hops = size; /* Selects one full ring traversal as the default number of exchanges. */

        /* Parse an optional hop count argument to allow experiments with partial or multiple traversals. */
        if (argc >= 2) {
            char *end = NULL;                 /* Holds parsing end pointer for validation. */
            long tmp = strtol(argv[1], &end, 10); /* Converts argv[1] to a long integer. */

            /* Validate that parsing consumed the full string and produced a non-negative hop count. */
            if (end == argv[1] || *end != '\0' || tmp < 0) {
                fprintf(stderr, "Usage: %s [hops]  (hops must be a non-negative integer)\n", argv[0]); /* Reports correct usage. */
                MPI_Abort(MPI_COMM_WORLD, 1); /* Aborts all ranks consistently on invalid input. */
            }

            hops = (int)tmp; /* Stores the validated hop count. */
        }
    }

    /* Broadcast hop count so all ranks execute the same number of exchanges. */
    MPI_Bcast(&hops, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Initialize the token only on rank 0 so the token has a single logical origin. */
    if (rank == 0) {
        token = 1; /* Sets a deterministic initial token value to simplify traceability. */
    } else {
        token = -1; /* Uses a sentinel value to emphasize that non-root ranks do not originate the token. */
    }

    /* Execute the ring exchanges for the configured number of hops. */
    for (int h = 0; h < hops; ++h) {
        int send_value = token; /* Captures the current token value as the send payload for this hop. */
        int recv_value = -1;    /* Provides storage for the token value received from the left neighbor. */

        /* Exchange token with neighbors using a deadlock-safe combined send/receive operation. */
        MPI_Sendrecv(&send_value, 1, MPI_INT, right, tag,
                     &recv_value, 1, MPI_INT, left, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Update local token to the value received from the left neighbor. */
        token = recv_value; /* Advances the token state by adopting the received value. */

        /* Optionally perturb the token to encode traversal; only rank 0 increments after each hop. */
        if (rank == 0) {
            token += 1; /* Modifies the token on rank 0 to make multi-hop progression observable. */
        }
    }

    /* Report the final token value on rank 0 to provide a single deterministic output point. */
    if (rank == 0) {
        printf("Ring completed: size=%d, hops=%d, final_token=%d\n", size, hops, token); /* Summarizes ring parameters and final token state. */
    }

    /* Finalize the MPI execution environment to release MPI resources cleanly. */
    MPI_Finalize();

    /* Return success status to the operating system. */
    return 0;
}
