/****************************************************************************/
/****************************************************************************/
/**                                                                        **/
/**                 TU München - Institut für Informatik                   **/
/**                                                                        **/
/** Copyright: Prof. Dr. Thomas Ludwig                                     **/
/**            Andreas C. Schmidt                                          **/
/**                                                                        **/
/** File:      partdiff.c                                                  **/
/**                                                                        **/
/** Purpose:   Partial differential equation solver for Gauß-Seidel and    **/
/**            Jacobi method.                                              **/
/**                                                                        **/
/****************************************************************************/
/****************************************************************************/

/* ************************************************************************ */
/* Include standard header file.                                            */
/* ************************************************************************ */
#define _POSIX_C_SOURCE 200809L

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff.h"

struct calculation_arguments
{
	uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
	uint64_t  num_matrices;   /* number of matrices                             */
	double    h;              /* length of a space between two lines            */
	double    ***Matrix;      /* index matrix used for addressing M             */
	double    *M;             /* two matrices with real values                  */
};

struct calculation_results
{
	uint64_t  m;
	uint64_t  stat_iteration; /* number of current iteration                    */
	double    stat_precision; /* actual precision of all slaves in iteration    */
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time;       /* time when program started                      */
struct timeval comp_time;        /* time when calculation completed                */


/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options)
{
	arguments->N = (options->interlines * 8) + 9 - 1;
	arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
	arguments->h = 1.0 / arguments->N;

	results->m = 0;
	results->stat_iteration = 0;
	results->stat_precision = 0;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static
void
freeMatrices (struct calculation_arguments* arguments)
{
	uint64_t i;

	for (i = 0; i < arguments->num_matrices; i++)
	{
		free(arguments->Matrix[i]);
	}

	free(arguments->Matrix);
	free(arguments->M);
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static
void*
allocateMemory (size_t size)
{
	void *p;

	if ((p = malloc(size)) == NULL)
	{
		printf("Speicherprobleme! (%" PRIu64 " Bytes angefordert)\n", size);
		exit(1);
	}

	return p;
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments)
{
	uint64_t i, j;

	uint64_t const N = arguments->N;

	arguments->M = allocateMemory(arguments->num_matrices * (N + 1) * (N + 1) * sizeof(double));
	arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

	for (i = 0; i < arguments->num_matrices; i++)
	{
		arguments->Matrix[i] = allocateMemory((N + 1) * sizeof(double*));

		for (j = 0; j <= N; j++)
		{
			arguments->Matrix[i][j] = arguments->M + (i * (N + 1) * (N + 1)) + (j * (N + 1));
		}
	}
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static
void
initMatrices (struct calculation_arguments* arguments, struct options const* options)
{
	uint64_t g, i, j;                                /*  local variables for loops   */

	uint64_t const N = arguments->N;
	double const h = arguments->h;
	double*** Matrix = arguments->Matrix;

	/* initialize matrix/matrices with zeros */
	for (g = 0; g < arguments->num_matrices; g++)
	{
		for (i = 0; i <= N; i++)
		{
			for (j = 0; j <= N; j++)
			{
				Matrix[g][i][j] = 0.0;
			}
		}
	}

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (options->inf_func == FUNC_F0)
	{
		for (g = 0; g < arguments->num_matrices; g++)
		{
			for (i = 0; i <= N; i++)
			{
				Matrix[g][i][0] = 1.0 - (h * i);
				Matrix[g][i][N] = h * i;
				Matrix[g][0][i] = 1.0 - (h * i);
				Matrix[g][N][i] = h * i;
			}

			Matrix[g][N][0] = 0.0;
			Matrix[g][0][N] = 0.0;
		}
	}
}

struct work_arguments 
{
	double **Matrix_In;
	double **Matrix_Out; 
	int start;
	int end;
	int N;
	double *maxresiduum_cache;
	int cache_index;
	double fpisin;
	double pih;
	int term_iteration;
	struct options const* options;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	pthread_t thread_id;
	int sub_wait;
	int main_wait;
};

static void* calculateRows(void* void_argument)
{
	struct work_arguments *argument = (struct work_arguments*) void_argument;

	double pih = argument->pih;
	double fpisin = argument->fpisin;
	int start = argument->start;
	int end = argument->end;
	int N = argument->N;
	double *maxresiduum_cache = argument->maxresiduum_cache;
	int cache_index = argument->cache_index;
	struct options const* options = argument->options;
	pthread_mutex_t *mutex = &argument->mutex;
	pthread_cond_t *cond = &argument->cond;

	int term_iteration;
	int i,j;
	double residuum, star;
	
	while (1)
	{
		// printf("locking on sub thread %d\n", cache_index);
		pthread_mutex_lock(mutex);
		
		while(argument->sub_wait)
		{
			// printf("waiting on sub thread %d\n", cache_index);
			pthread_cond_wait(cond, mutex);
		}

		term_iteration = argument->term_iteration;

		if (term_iteration <= 0)
		{
			break;
		}
		
		// printf("calculating on sub thread %d\n", cache_index);

		argument->sub_wait=1;

		double **Matrix_In = argument->Matrix_In;
		double **Matrix_Out = argument->Matrix_Out;

		double maxresiduum = 0;
	
		/* over rows inclusive 'start' to exclusive 'end' */
		for (i = start; i < end; i++)
		{
			double fpisin_i = 0.0;

			if (options->inf_func == FUNC_FPISIN)
			{
				fpisin_i = fpisin * sin(pih * (double)i);
			}

			/* over all columns */
			for (j = 1; j < N; j++)
			{
				star = 0.25 * (Matrix_In[i-1][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + Matrix_In[i+1][j]);

				if (options->inf_func == FUNC_FPISIN)
				{
					star += fpisin_i * sin(pih * (double)j);
				}

				if (options->termination == TERM_PREC || term_iteration == 1)
				{
					residuum = Matrix_In[i][j] - star;
					residuum = (residuum < 0) ? -residuum : residuum;
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}

				Matrix_Out[i][j] = star;
			}
		}
		maxresiduum_cache[cache_index] = maxresiduum;


		// printf("right before signaling on sub thread %d\n", cache_index);
		argument->main_wait = 0;
		pthread_cond_signal(cond);
		pthread_mutex_unlock(mutex);
		// printf("signaling on sub thread %d\n", cache_index);
	}

	argument->main_wait = 0;
	pthread_cond_signal(cond);
	pthread_mutex_unlock(mutex);
	printf("Finished Thread %d\n", cache_index);
	return 0;
}

/* ************************************************************************ */
/* calculate: solves the equation                                           */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
	int i;                                      /* local variables for loops */
	int m1, m2;                                 /* used as indices for old and new matrices */
	double maxresiduum;                         /* maximum residuum value of a slave in iteration */

	int const N = arguments->N;
	double const h = arguments->h;

	double pih = 0.0;
	double fpisin = 0.0;

	int term_iteration = options->term_iteration;
	int num_threads = options->number;

	if (num_threads <= 0)
	{
		num_threads = 1;
	}

	/* initialize m1 and m2 depending on algorithm */
	if (options->method == METH_JACOBI)
	{
		m1 = 0;
		m2 = 1;
	}
	else
	{
		m1 = 0;
		m2 = 0;
	}

	if (options->inf_func == FUNC_FPISIN)
	{
		pih = PI * h;
		fpisin = 0.25 * TWO_PI_SQUARE * h * h;
	}

	double *maxresiduum_cache = allocateMemory(num_threads * sizeof(double));

	for (i = 0; i < num_threads; i++)
	{
		maxresiduum_cache[i] = 0;
	}

	struct work_arguments *args = allocateMemory(num_threads * sizeof(struct work_arguments));
	pthread_t *thread_ids = allocateMemory(num_threads * sizeof(pthread_t)); 

	int chunkSize = N / num_threads;

	printf("hello on main thread\n");
	
	for (i = 0; i < num_threads; i++)
	{
		struct work_arguments *work_argument = &args[i];
		int start = (chunkSize * i) + 1;
		int end = start + chunkSize;
		
		if ((i+1) >= num_threads)
		{
			end = N;
		}

		work_argument->start = start;
		work_argument->end = end;
		work_argument->cache_index = i;
		work_argument->fpisin = fpisin;
		work_argument->maxresiduum_cache = maxresiduum_cache;
		work_argument->pih = pih;
		work_argument->options = options;
		work_argument->N = N;
		work_argument->sub_wait = 1;
		work_argument->main_wait = 1;
		work_argument->term_iteration = term_iteration;
		
		pthread_mutex_t *mutex = &work_argument->mutex;
		pthread_cond_t *cond = &work_argument->cond;

		pthread_mutex_init(mutex, NULL);
		pthread_cond_init(cond, NULL);
		printf("initial locking on main thread for %d\n", i);
		pthread_mutex_lock(mutex);

		printf("creating sub thread %d\n", i);
		if(pthread_create(&work_argument->thread_id, NULL, calculateRows, work_argument))
		{
			fprintf(stderr, "Error creating thread %d\n", i);
			exit(1);
		}
	}
	

	while (term_iteration > 0)
	{
		printf("Iterations left: %d -------------------------------\n", term_iteration);

		double** Matrix_Out = arguments->Matrix[m1];
		double** Matrix_In  = arguments->Matrix[m2];
		
		for (i = 0; i < num_threads; i++)
		{
			struct work_arguments *argument = &args[i];
			argument->Matrix_In = Matrix_In;
			argument->Matrix_Out = Matrix_Out;
			argument->term_iteration = term_iteration;

			printf("unlocking on main thread for %d\n", i);
			argument->sub_wait = 0;
			argument->main_wait = 1;

			pthread_cond_signal(&argument->cond);
			pthread_mutex_unlock(&argument->mutex);
		}

		for (i = 0; i < num_threads; i++)
		{
			struct work_arguments *argument = &args[i];
			
			printf("locking for waiting on main thread for %d\n", i);
			pthread_mutex_lock(&argument->mutex);

			while (argument->main_wait)
			{
				printf("before waiting got lock for mutex of thread %d\n", i);
				pthread_cond_wait(&argument->cond, &argument->mutex);
				printf("after waiting got lock for mutex of thread %d\n", i);
			}
			printf("got finished signal for thread %d\n", i);
		}

		maxresiduum = 0;

		if (options->termination == TERM_PREC || term_iteration == 1)
		{
			for (i = 0; i < num_threads; i++)
			{
				double thread_max = maxresiduum_cache[i];
				maxresiduum = (thread_max < maxresiduum) ? maxresiduum : thread_max;
				maxresiduum_cache[i] = 0;
			}
		}

		results->stat_iteration++;
		results->stat_precision = maxresiduum;

		/* exchange m1 and m2 */
		i = m1;
		m1 = m2;
		m2 = i;

		/* check for stopping calculation depending on termination method */
		if (options->termination == TERM_PREC)
		{
			if (maxresiduum < options->term_precision)
			{
				term_iteration = 0;
			}
		}
		else if (options->termination == TERM_ITER)
		{
			term_iteration--;
		}
	}

	printf("Finished on main-------------------------\n");
	for (i = 0; i < num_threads; i++)
	{
		struct work_arguments *argument = &args[i];
		argument->sub_wait = 0;
		argument->term_iteration = 0;
		pthread_cond_signal(&argument->cond);
		pthread_mutex_unlock(&argument->mutex);
		pthread_join(argument->thread_id, NULL);
		pthread_mutex_destroy(&argument->mutex);
		pthread_cond_destroy(&argument->cond);
	}
	free(thread_ids);
	free(args);
	free(maxresiduum_cache);

	results->m = m2;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static
void
displayStatistics (struct calculation_arguments const* arguments, struct calculation_results const* results, struct options const* options)
{
	int N = arguments->N;
	double time = (comp_time.tv_sec - start_time.tv_sec) + (comp_time.tv_usec - start_time.tv_usec) * 1e-6;

	printf("Berechnungszeit:    %f s \n", time);
	printf("Speicherbedarf:     %f MiB\n", (N + 1) * (N + 1) * sizeof(double) * arguments->num_matrices / 1024.0 / 1024.0);
	printf("Berechnungsmethode: ");

	if (options->method == METH_GAUSS_SEIDEL)
	{
		printf("Gauß-Seidel");
	}
	else if (options->method == METH_JACOBI)
	{
		printf("Jacobi");
	}

	printf("\n");
	printf("Interlines:         %" PRIu64 "\n",options->interlines);
	printf("Stoerfunktion:      ");

	if (options->inf_func == FUNC_F0)
	{
		printf("f(x,y) = 0");
	}
	else if (options->inf_func == FUNC_FPISIN)
	{
		printf("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)");
	}

	printf("\n");
	printf("Terminierung:       ");

	if (options->termination == TERM_PREC)
	{
		printf("Hinreichende Genaugkeit");
	}
	else if (options->termination == TERM_ITER)
	{
		printf("Anzahl der Iterationen");
	}

	printf("\n");
	printf("Anzahl Iterationen: %" PRIu64 "\n", results->stat_iteration);
	printf("Norm des Fehlers:   %e\n", results->stat_precision);
	printf("\n");
}

/****************************************************************************/
/** Beschreibung der Funktion displayMatrix:                               **/
/**                                                                        **/
/** Die Funktion displayMatrix gibt eine Matrix                            **/
/** in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.   **/
/**                                                                        **/
/** Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix    **/
/** ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie   **/
/** sieben Zwischenzeilen ausgegeben.                                      **/
/****************************************************************************/
static
void
displayMatrix (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
{
	int x, y;

	double** Matrix = arguments->Matrix[results->m];

	int const interlines = options->interlines;

	printf("Matrix:\n");

	for (y = 0; y < 9; y++)
	{
		for (x = 0; x < 9; x++)
		{
			printf ("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
		}

		printf ("\n");
	}

	fflush (stdout);
}

/* ************************************************************************ */
/*  main                                                                    */
/* ************************************************************************ */
int
main (int argc, char** argv)
{
	struct options options;
	struct calculation_arguments arguments;
	struct calculation_results results;

	askParams(&options, argc, argv);

	initVariables(&arguments, &results, &options);

	allocateMatrices(&arguments);
	initMatrices(&arguments, &options);

	gettimeofday(&start_time, NULL);
	calculate(&arguments, &results, &options);
	gettimeofday(&comp_time, NULL);

	displayStatistics(&arguments, &results, &options);
	displayMatrix(&arguments, &results, &options);

	freeMatrices(&arguments);

	return 0;
}
