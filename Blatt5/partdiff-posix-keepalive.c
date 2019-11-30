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

/*
    Argument Struct for Worker Threads.
    Its values will possible change
    while a worker thread has access to
    an instance.
 */
struct work_arguments 
{
    double **Matrix_In;            /* Input Matrix                                                            */
    double **Matrix_Out;           /* Output Matrix                                                           */
    int start;                     /* Start Index for rows                                                    */
    int end;                       /* End Index for rows                                                      */
    int N;                         /* number of spaces between lines (lines=N+1)                              */
    double *maxresiduum_cache;     /* Array for maxresiduum results, will be read after each Iteration        */
    int cache_index;               /* The unique index for the worker in max_residuum_cache                   */
    double fpisin;                 /* Precalculated Value: '0.25 * TWO_PI_SQUARE * h * h' or zero             */
    double pih;                    /* Precalculated Value: 'PI * h' or zero                                   */
    int term_iteration;            /* Current Iteration                                                       */
    struct options const* options; /* Options for this Run                                                    */
    pthread_mutex_t mutex;         /* Mutex for the worker to synchronize the execution with main thread      */
    pthread_cond_t cond;           /* Conditional Variable to signal the execution of worker and main thread  */
    pthread_t thread_id;           /* ThreadId of the Worker thread                                           */
    int sub_wait;                  /* Variable for signaling if the worker thread should wait, either 0 or 1  */
    int main_wait;                 /* Variable for signaling if if the main thread should wait, either 0 or 1 */
};

/*
    Function which will be called on Worker Thread.
    Needs to have a single Parameter with void* as type and
    needs to return a void Pointer, commonly 0.
 */
static void* calculateRows(void* void_argument)
{
    struct work_arguments *argument = (struct work_arguments*) void_argument;

    /* Extract most variables of work_arguments for faster/better access */
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

    /*
        loop forever, as it should not check for term_iteration
        at this point, else it could lead to race conditions
     */
    while (1)
    {
        /*
            always lock before calling pthread_cond_wait
            as it may not block instead if not locked
            by this thread.
        */
        pthread_mutex_lock(mutex);

        /*
            Use a while instead of if in case the thread is scheduled weirdly which
            could cause weird effects (recommendation of pthread documentation).
            signal variable for sub thread to wait.
            Only the main thread should change this value to zero
         */
        while(argument->sub_wait)
        {
            /*
                wait for signal from main thread for cond on mutex,
                waiting/sleeping releases the previously held lock and locks
                instantly (atomic) the moment it receives a signal
                via cond
            */
            pthread_cond_wait(cond, mutex);
        }

        /*
            always update term_iteration AFTER awakening to prevent
            race conditions, as the main thread does not manipulate
            this values when it does not hold the mutex lock.
         */
        term_iteration = argument->term_iteration;

        /*
            check after each awakening if the termination condition
            for worker (and main thread) is reached, so it can
            terminate successfully and be cleaned up
         */
        if (term_iteration <= 0)
        {
            break;
        }
        /*
            set sub_wait to 1 in worker thread, as main thread
            cannot (should not) set this value while the worker
            calculates, as it could possibly lead to race conditions
            on main thread writing to sub_wait and worker thread reading
            on sub_wait
         */
        argument->sub_wait=1;

        /* get the current values of Matrix_In and Matrix_Out */
        double **Matrix_In = argument->Matrix_In;
        double **Matrix_Out = argument->Matrix_Out;

        double maxresiduum = 0;

        /*
            As no synchronization needs to be done between Main Thread and worker thread
            this calculataion can be copied completely.
            It does not need synchronization, as the only changed value is
            an element of rows specific to this worker of Matrix_Out, 
            which is not updated by any other thread.
            Only the start value and end value for i needs to be replaced with
            the 'start' and 'end' variable respectively, as it iterates not
            over all rows, but only a portion of it.
         */
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
        /*
            As this function should not return a value like maxresiduum with 'return'
            it needs a output variable like maxresiduum_cache, where it is stored
            and then read from the Main Thread.
        */
        maxresiduum_cache[cache_index] = maxresiduum;

        /*
            set main_wait to 0 in worker thread, as main thread
            cannot (should not be able to) set this value to zero, 
            while the worker calculates, as it should wait on the
            condition variable 'cond'.
            After that this worker thread will signal the main thread
            that it finished it's work with pthread_cond_signal.
            After signaling the mutex, which this worker thread holds,
            needs to be unlocked, so that the main thread can
            awake from waiting and acquire the mutex
         */
        argument->main_wait = 0;
        pthread_cond_signal(cond);
        pthread_mutex_unlock(mutex);
    }

    /*
        Nearly the same as at the end of the previous while
        loop. Here it signals the main thread a last time, so
        main thread is able to acquire the lock in any case:
        while loop should only finish when term_iteration is
        below or equal to zero, thus for the main thread to
        destroy all resources (mutex, condition variables)
        successfully, it needs to acquire the the mutex which
        is held by worker thread after the while loop exits
    */
    argument->main_wait = 0;
    pthread_cond_signal(cond);
    pthread_mutex_unlock(mutex);
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

    /* Array where the output of maxresiduum is stored after for each worker */
    double *maxresiduum_cache = allocateMemory(num_threads * sizeof(double));

    /* initialize maxresiduum_cache entries to zero */
    for (i = 0; i < num_threads; i++)
    {
        maxresiduum_cache[i] = 0;
    }

    /* allocate array for workarguments, where its values are stored in_place for faster access */
    struct work_arguments *args = allocateMemory(num_threads * sizeof(struct work_arguments));

    /* allocate array for threadids with value stored in-place */
    pthread_t *thread_ids = allocateMemory(num_threads * sizeof(pthread_t)); 

    int chunkSize = N / num_threads;
    
    /* 
        create and initialize worker threads and their work_argument in this loop before 
        the actual calculation loop and store their thread_ids for further thread control
     */
    for (i = 0; i < num_threads; i++)
    {
        /*
            a struct pointer is needed to modify the values of its content,
            else a copy is constructed and modified instead
         */
        struct work_arguments *work_argument = &args[i];

        /*
            start variable is incremented, as it is done in the previous outer loop.
            the calculation uses the elements adjacent to the current element
            and thus it would access a value out of its bounds if it starts from zero.
        */
        int start = (chunkSize * i) + 1;
        int end = start + chunkSize;

        /*
            necessary check to prevent accessing out of the bounds of the Matrix,
            as the last work_chunk goes to N + 1 instead of N 
            as start is incremented by one.
            This also means that the worker for the last chunk in the current iteration 
            always gets one row less than the others
         */
        if ((i+1) >= num_threads)
        {
            end = N;
        }

        /* initialize work_argument values */
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

        /*
            Mutexes and conditional variables are part of the workargument instead of a
            pointer for faster access.
            Each worker thread gets a mutex, so that they don't exclude each other from
            working. The main thread is responsible for synchronizing the work on each
            worker. Each worker receive a conditional variable for signaling the start
            and end of each work cycle of the worker.
         */
        pthread_mutex_t *mutex = &work_argument->mutex;
        pthread_cond_t *cond = &work_argument->cond;

        /* Initialize the mutex and conditional variable with default values */
        pthread_mutex_init(mutex, NULL);
        pthread_cond_init(cond, NULL);
        /* acquire mutex on main thread before worker thread is created */
        pthread_mutex_lock(mutex);

        /* 
            create pthread worker with function calculate and its argument and store its id in thread_ids.
            pthread_create can return a value different than zero to indicate a specific error type,
            in our case, we just terminate the programm if we cant create a worker thread.
        */
        if(pthread_create(&work_argument->thread_id, NULL, calculateRows, work_argument))
        {
            fprintf(stderr, "Error creating thread %d\n", i);
            exit(1);
        }
    }
    

    /* the actual calculation loop */
    while (term_iteration > 0)
    {
        double** Matrix_Out = arguments->Matrix[m1];
        double** Matrix_In  = arguments->Matrix[m2];

        for (i = 0; i < num_threads; i++)
        {
            /*
                these values needs to be updated before each work cycle
                as they change possible each iteration
             */
            struct work_arguments *argument = &args[i];
            argument->Matrix_In = Matrix_In;
            argument->Matrix_Out = Matrix_Out;
            argument->term_iteration = term_iteration;

            /*
                set sub_wait to zero before signaling worker thread
                that it can continue execution.
             */
            argument->sub_wait = 0;
            /*
                set main_wait to 1, which indicates that the main thread
                shall wait till it is set to 0 from a worker thread
             */
            argument->main_wait = 1;

            pthread_cond_signal(&argument->cond);
            /*
                For the worker thread
                to actually execute again, the main thread needs to
                release the ownership of the mutex for the worker.
            */
            pthread_mutex_unlock(&argument->mutex);
        }

        for (i = 0; i < num_threads; i++)
        {
            struct work_arguments *argument = &args[i];

            /*
                always lock before calling pthread_cond_wait
                as it may not block instead if not locked
                by this thread.
            */
            pthread_mutex_lock(&argument->mutex);

            /*
                use a while instead of if in case the thread is scheduled weirdly which
                could cause weird effects (recommendation of pthread documentation)
             */
            while (argument->main_wait)
            {
                /*
                    wait for signal from worker thread for cond on mutex,
                    waiting/sleeping releases the previously held lock and locks
                    instantly (atomic) the moment it receives a signal
                    via cond
                */
                pthread_cond_wait(&argument->cond, &argument->mutex);
            }
        }

        maxresiduum = 0;

        if (options->termination == TERM_PREC || term_iteration == 1)
        {
            /*
                combine the results for maxresiduum of each worker.
                it resembles "reduction(max, maxresiduum)" of openmp
            */
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

    /*
        Signal and wait for worker threads to terminate.
        Cleanup the mutex and conditional variables.
     */
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

    /* free previously in this function dynamically allocated memory */
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
