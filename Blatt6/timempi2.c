#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

int main()
{
int length = 300;

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  char out[length];
  struct timeval tv;
  gettimeofday(&tv, NULL);

  if(world_rank == 0)
    {
      tv.tv_usec = 999999;
    }

  if (world_rank != 0)
    {
      char hostname[200];
      gethostname(hostname, sizeof(hostname));

      char tmbuf[64];
      struct tm *nowtm;

      time_t nowtime = tv.tv_sec;
      nowtm = localtime(&nowtime);
      strftime(tmbuf, sizeof tmbuf, "%Y-%m-%d %H:%M:%S", nowtm);
      snprintf(out, sizeof out, "%s: %s.%06ld", hostname, tmbuf, tv.tv_usec);

      MPI_Send(&out, length, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

  if (world_rank == 0 && world_size >= 2)
    {
        for (int i = 1; i < world_size ; i++)
        {
          MPI_Recv(&out, length, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("%s\n", out);
        }
    }
  long min;

  MPI_Reduce(&tv.tv_usec, &min, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

  if(world_rank == 0)
  {
    printf("%ld\n", min);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  printf("Rang %d beendet jetzt\n", world_rank);

  // Finalize the MPI environment.
  MPI_Finalize();
}
