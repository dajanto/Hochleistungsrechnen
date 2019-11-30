#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>

int signal = 1; //Konstante die beim Senden zur Signalisierung verwendet wird

/*
Initialisierung von nprocs Arrays (in jedem Prozess ein Array). Wenn die Länge
nicht ganz aufgeht bekommen die ersten Arrays einen Eintrag mehr bis man die
geforderte Anzahl errreicht hat. Die restlichen Werte werden mit -1 ungültig gesetzt.
*/
int*
init (int N, int rank, int nprocs, int base, int length)
{
	// TODO
	int rest = N % nprocs;

	int* buf = malloc(sizeof(int) * length);

	srand(time(NULL) + rank);

	if(rank < rest)
	{
		for(int i = 0; i < length; i++)
		{
			buf[i] = rand() % 25;
		}
	}
	else
	{
		for(int i = 0; i < base; i++)
		{
			buf[i] = rand() % 25;
		}
		buf[base] = -1;
	}

	return buf;
}

/*
Hilfsfunktion von cycle sendet Array-Einträge einen Prozess weiter
*/
void
rotate(int* buf,int rank, int nprocs, int length)
{
	if (rank == 0)
	{
		MPI_Send(buf, length, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(buf, length, MPI_INT, nprocs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	else
	{
		int* buf_copy = malloc(sizeof(int) * length);

		for (int i = 0; i < length; i++)
		{
			buf_copy[i] = buf[i];
		}

		MPI_Recv(buf, length, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(buf_copy, length, MPI_INT, (rank + 1) % nprocs, 0, MPI_COMM_WORLD);
	}
}

/*
Für das "Drehen" verantworlich, die Einträge werden weitergeschickt
bis der Wert, des ersten Eintrags des ersten Arrays beim ersten Eintrags
des letzten Arrays angekommen ist.
*/
int*
circle (int* buf, int rank, int nprocs, int length)
{
	int goal = -1;
	if(rank == 0)
	{
		goal = buf[0];
		MPI_Send(&goal, 1, MPI_INT, nprocs - 1, 0, MPI_COMM_WORLD);
	}

	if(rank == nprocs -1)
	{
		MPI_Recv(&goal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}


	int status;

	do
	{
		rotate(buf, rank, nprocs, length);

		if (goal == buf[0])
		{
			status = 1;
		}
		else
		{
			status = 0;
		}
		MPI_Bcast(&status, 1, MPI_INT, nprocs -1, MPI_COMM_WORLD);
	}
	while(status == 0);


	return buf;
}

/*
Für das Drucken verantworlich die Prozesse drucken nacheinander das Array,
was sie halten.
*/
void print_processes(int rank, int* buf, int basis, int nprocs)
{
	int base = basis;

	if (rank == 0)
	{
    MPI_Send(&signal, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    for (int i = 0; i < base; i++)
		{
			printf("rank %d: %d\n", rank, buf[i]);
		}
		if (buf[base] != -1)
		{
			printf("rank %d: %d\n", rank, buf[base]);
		}
	}
	else
	{
    int buffer;
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &buffer);
    if (buffer == 1)
		{
      MPI_Recv(&signal, buffer, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			for (int i = 0; i < base; i++)
			{
				printf("rank %d: %d\n", rank, buf[i]);
			}
			if (buf[base] != -1)
			{
				printf("rank %d: %d\n", rank, buf[base]);
			}
      if (rank + 1 != nprocs)
			{
				MPI_Send(&signal, 1, MPI_INT, ++rank, 0, MPI_COMM_WORLD);
    	}
  	}
	}
}

int
main (int argc, char** argv)
{
	int N;
	int rank;
	int* buf;
	int nprocs;



	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// Get the rank of the process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	if (argc < 2)
	{
		printf("Arguments error!\nPlease specify a buffer size.\n");
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// Array length
	N = atoi(argv[1]);

//Für den Fall, dass der Nutzer mehr Prozesse anfordert, als er Einträge im Array hat.
	if(N < nprocs)
	{
		printf("Arguments error!\nPlease don't use more processes than Array-Entries.\n");
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	int base = N / nprocs;
	int length = base + 1;

	buf = init(N, rank, nprocs, base, length);

	if (nprocs > 1)
	{
		if (rank == 0)
		{
			printf("\nBEFORE\n");
		}

		print_processes(rank, buf, base, nprocs);

//Ich bin mir nicht sicher ob die Barriere notwendig ist
		MPI_Barrier(MPI_COMM_WORLD);

		circle(buf, rank, nprocs, length);

		if (rank == 0)
		{
			printf("\nAFTER\n");
		}

		print_processes(rank, buf, base, nprocs);
	}
	else
	{
		printf("\nBEFORE\n");

		for (int i = 0; i < N; i++)
		{
			printf("rank %d: %d\n", rank, buf[i]);
		}

		printf("\nAFTER\n");

		for (int j = 0; j < N; j++)
		{
			printf("rank %d: %d\n", rank, buf[j]);
		}

	}


	return EXIT_SUCCESS;
}
