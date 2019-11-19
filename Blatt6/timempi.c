#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
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

	strftime(buffer,30,"%d.%m.%Y %T.",localtime(&curtime));
	char name[150];
	memset(name, 0, 150);
	gethostname(name, 150);
    
	char *responses[world_size];

	char message[255];
	sprintf(message, "%s: %s%ld\n", name, buffer, tv.tv_usec);

	if (world_rank == 0) {
		for (int i = 1; i < world_size; i++){
			MPI_Recv(&responses[i], 255, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	} else {
		char message[255];
		sprintf(message, "%s: %s%ld\n", name, buffer, tv.tv_usec);
		MPI_Send(message, 255, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
	}
	for (int i = 1; i < world_size; i++)
	{
		printf("Message from Rank %d: %s", i, responses[i]);
	}

	printf("Rang %d beendet jetzt!\n", world_rank);
    // Finalize the MPI environment.
    MPI_Finalize();
}