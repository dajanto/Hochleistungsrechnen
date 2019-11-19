#define _DEFAULT_SOURCE

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        
    char buffer[30];
    struct timeval tv;

    time_t curtime;

    gettimeofday(&tv, NULL); 
    curtime=tv.tv_sec;

    // format time_val struct into a c string and put it into buffer
    strftime(buffer,30,"%d.%m.%Y %T.",localtime(&curtime));
    char name[150];

    gethostname(name, 150);
    
    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++){
            char message[255];
            // wait for message of process of rank i with a maximum of 255 chars (mpi_bytes)
            MPI_Recv(message, 255, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", message);
        }
    } else {
        char message[255];
        sprintf(message, "%s %d: %s%ld", name, world_rank, buffer, tv.tv_usec);
        MPI_Send(message, 255, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    // only rank 0 gets the result, everything else gets 'garbage' (nothing)
    long result;

    MPI_Reduce(&tv.tv_usec, &result, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

    if (world_rank == 0){
        printf("Kleinster Mikrosekunden Anteil: %ld\n", result);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rang %d beendet jetzt!\n", world_rank);
    // Finalize the MPI environment.
    MPI_Finalize();
}