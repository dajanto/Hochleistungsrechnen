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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff.h"
#include <mpi.h>
#include <omp.h>

struct calculation_arguments
{
    uint64_t  N;              /* number of spaces between lines (lines=N+1)     */
    uint64_t  num_matrices;   /* number of matrices                             */
    double    h;              /* length of a space between two lines            */
    double    ***Matrix;      /* index matrix used for addressing M             */
    double    *M;             /* two matrices with real values                  */
    uint64_t  chunkSize;
    uint64_t  chunkStart;
    uint64_t  chunkEnd;
    int  rank;
    int  nprocs;
    double  **cache;
    double  *cache_line;
    uint64_t  *indexTable;
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

static
int
useMpi_Gauss(int method, int nprocs)
{
    return method == METH_GAUSS_SEIDEL && nprocs > 1;
}

/* ************************************************************************ */
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options, int rank, int nprocs)
{
    arguments->N = (options->interlines * 8) + 9 - 1;
    arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
    arguments->h = 1.0 / arguments->N;
    results->m = 0;
    results->stat_iteration = 0;
    results->stat_precision = 0;

    uint64_t chunkSize = (arguments->N - 1) / nprocs;
    uint64_t start = (chunkSize * rank) + 1;
    uint64_t end = start + chunkSize - 1;

    if ((rank+1) >= nprocs)
    {
        chunkSize = chunkSize + (arguments->N - end) - 1;
        end = arguments->N - 1;
    }

    chunkSize++;
    arguments->chunkSize = chunkSize;
    arguments->chunkStart = start;
    arguments->chunkEnd = end;
    arguments->rank = rank;
    arguments->nprocs = nprocs;

    if (useMpi_Gauss(options->method, arguments->nprocs))
    {
        arguments->indexTable = allocateMemory((chunkSize + 1) * sizeof(uint64_t));
        uint64_t i;

        for (i = 0; i <= chunkSize; i++)
        {
            arguments->indexTable[i] = (i * nprocs) + rank;
        }
    }else
    {
        arguments->indexTable = NULL;
    }
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

    if (arguments->cache)
    {
        free(arguments->cache);
    }

    if (arguments->cache_line)
    {
        free(arguments->cache_line);
    }
    if (arguments->indexTable)
    {
        free(arguments->indexTable);
    }
    free(arguments->Matrix);
    free(arguments->M);
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments, struct options* options)
{
    uint64_t i, j;

    uint64_t const N = arguments->N;
    uint64_t const chunkSize = arguments->chunkSize;

    arguments->M = allocateMemory(arguments->num_matrices * (chunkSize + 1) * (N + 1) * sizeof(double));
    arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

    for (i = 0; i < arguments->num_matrices; i++)
    {
        arguments->Matrix[i] = allocateMemory((chunkSize + 1) * sizeof(double*));

        for (j = 0; j <= chunkSize; j++)
        {
            arguments->Matrix[i][j] = arguments->M + (i * (chunkSize + 1) * (N + 1)) + (j * (N + 1));
        }
    }
    if (useMpi_Gauss(options->method, arguments->nprocs))
    {
        arguments->cache_line = allocateMemory((chunkSize + 1) * (N + 1) * 2 * sizeof(double));
        
        arguments->cache = allocateMemory((chunkSize + 1) * 2 * sizeof(double*));

        /* map start of rows of cache to start of rows in linear matrix cache_line (todo: is chunkSize + 1 valid?) */
        for (j = 0; j <= ((chunkSize+1) * 2); j++)
        {
            arguments->cache[j] = arguments->cache_line + (j * (N + 1));
        }
    }
    else
    {
        arguments->cache = NULL;
        arguments->cache_line = NULL;
    }
    
}

static
uint64_t
lineToIndex(const struct calculation_arguments* arguments, const struct options* options, uint64_t line)
{
    if (useMpi_Gauss(options->method, arguments->nprocs))
    {
        for (uint64_t i = 0; i <= (arguments->chunkSize * 2); i++)
        {
            uint64_t value = arguments->indexTable[i];
            if (value == line){
                return i;
            }
        }
        printf("Invalid Index Map, Process %d does not own Line %ld", arguments->rank, line);
        exit(1);
    }
    else
    {
        return line - arguments->chunkStart + 1;
    }
}

static
uint64_t
indexToLine(const struct calculation_arguments* arguments, const struct options* options, uint64_t index)
{
    if (useMpi_Gauss(options->method, arguments->nprocs))
    {
        return arguments->indexTable[index];
    }
    else
    {
        return index + arguments->chunkStart;
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
    uint64_t const chunkSize = arguments->chunkSize;
    double const h = arguments->h;
    double*** Matrix = arguments->Matrix;

    /* initialize matrix/matrices with zeros */
    for (g = 0; g < arguments->num_matrices; g++)
    {
        for (i = 0; i <= chunkSize; i++)
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
            for (i = 0; i <= chunkSize; i++)
            {
                uint64_t line = indexToLine(arguments, options, i);

                Matrix[g][line][0] = 1.0 - (h * line);
                Matrix[g][line][N] = h * line;
                Matrix[g][0][i] = 1.0 - (h * line);
                Matrix[g][N][i] = h * line;
            }

            Matrix[g][chunkSize][0] = 0.0;
            Matrix[g][0][N] = 0.0;
        }
    }
}

/* ************************************************************************ */
/* calculate: solves the equation sequentially for either jacobi or gauß    */
/* ************************************************************************ */
static
void
calculate (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
    uint64_t i, j;                                   /* local variables for loops */
    int m1, m2;                                 /* used as indices for old and new matrices */
    double star;                                /* four times center value minus 4 neigh.b values */
    double residuum;                            /* residuum of current iteration */
    double maxresiduum;                         /* maximum residuum value of a slave in iteration */

    uint64_t const N = arguments->N;
    double const h = arguments->h;

    double pih = 0.0;
    double fpisin = 0.0;

    int term_iteration = options->term_iteration;

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

    while (term_iteration > 0)
    {
        double** Matrix_Out = arguments->Matrix[m1];
        double** Matrix_In  = arguments->Matrix[m2];

        maxresiduum = 0;

        /* over all rows */
        for (i = 1; i < N; i++)
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

    results->m = m2;
}

/* ************************************************************************ */
/* calculate: solves the equation of gauß seidel in parallel with mpi only  */
/* ************************************************************************ */
static
void
calculate_mpi_gseidel (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
    uint64_t i, j;                              /* local variables for loops */
    int m1, m2;                                 /* used as indices for old and new matrices */
    double star;                                /* four times center value minus 4 neigh.b values */
    double residuum;                            /* residuum of current iteration */
    double maxresiduum;                         /* maximum residuum value of a slave in iteration */

    uint64_t const N = arguments->N;
    double const h = arguments->h;

    uint64_t *indexTable = arguments->indexTable;
    double **cache = arguments->cache;
    uint64_t chunkSize = arguments->chunkSize;
    uint64_t rank = arguments->rank;
    uint64_t nprocs = arguments->nprocs;
    int notFirstRank = rank > 0;
    int notLastRank = rank < (nprocs - 1);
    uint64_t chunkStart;
    uint64_t chunkEnd;
    uint64_t cache_i;

    double pih = 0.0;
    double fpisin = 0.0;

    int term_iteration = options->term_iteration;

    /* as this algorithm is exclusive for gauß seidel, no check is needed */
    m1 = 0;
    m2 = 0;

    if (options->inf_func == FUNC_FPISIN)
    {
        pih = PI * h;
        fpisin = 0.25 * TWO_PI_SQUARE * h * h;
    }

    while (term_iteration > 0)
    {
        double** Matrix_Out = arguments->Matrix[m1];
        double** Matrix_In  = arguments->Matrix[m2];

        maxresiduum = 0;
        chunkStart = 0;
        chunkEnd = chunkSize;

        /* over all rows */
        for (i = 1, cache_i = 0; i < chunkSize; i++, cache_i += 2)
        {
            double fpisin_i = 0.0;

            if (options->inf_func == FUNC_FPISIN)
            {
                fpisin_i = fpisin * sin(pih * (double)indexTable[i]);
            }
            // printf("i: %ld, cache_i: %ld, chunkSize: %ld\n", i, cache_i, chunkSize);
            /* over all columns */
            for (j = 1; j < N; j++)
            {
                /* if a new chunk was just reached, wait for the previous rank if available to send chunk */
                if (chunkStart == j && notFirstRank)
                {
                    MPI_Recv(cache[cache_i] + chunkStart, chunkSize, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                star = 0.25 * (cache[cache_i][j] + Matrix_In[i][j-1] + Matrix_In[i][j+1] + cache[cache_i + 1][j]);

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

                /* if reached the end of the chunk or the last column, send it to the next rank if available */
                if ((chunkEnd <= j || j == (N - 1)) && notLastRank)
                {
                    MPI_Send(Matrix_Out[i] + chunkStart, chunkSize, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                    chunkStart += chunkSize;
                    chunkEnd += chunkSize;
                }
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, &maxresiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
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

    results->m = m2;
}

/* ************************************************************************ */
/* calculate: solves the equation with jacobi in paralel with open mp only  */
/* ************************************************************************ */
static
void
calculate_omp (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
    uint64_t i, j;                                   /* local variables for loops */
    int m1, m2;                                 /* used as indices for old and new matrices */
    double star;                                /* four times center value minus 4 neigh.b values */
    double residuum;                            /* residuum of current iteration */
    double maxresiduum;                         /* maximum residuum value of a slave in iteration */

    uint64_t const N = arguments->N;
    double const h = arguments->h;

    double pih = 0.0;
    double fpisin = 0.0;

    int term_iteration = options->term_iteration;

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

    while (term_iteration > 0)
    {
        double** Matrix_Out = arguments->Matrix[m1];
        double** Matrix_In  = arguments->Matrix[m2];

        maxresiduum = 0;

        #pragma omp parallel for private(i, j, star, residuum) reduction(max: maxresiduum)
        /* over all rows */
        for (i = 1; i < N; i++)
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

    results->m = m2;
}

/* ************************************************************************ */
/* calculate: solves the equation with jacobi in parallel with omp and mpi  */
/* ************************************************************************ */
static
void
calculate_mpi_jacobi (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
    uint64_t i, j;                                   /* local variables for loops */
    int m1, m2;                                 /* used as indices for old and new matrices */
    double star;                                /* four times center value minus 4 neigh.b values */
    double residuum;                            /* residuum of current iteration */
    double maxresiduum;                         /* maximum residuum value of a slave in iteration */
    double recvmaxresiduum;                     /* tmp holder for maximum residuum value of global in iteration */

    uint64_t const N = arguments->N;
    double const h = arguments->h;
    int chunkSize = arguments->chunkSize;
    int rank = arguments->rank;
    int lastRank = arguments->nprocs - 1;
    int rowOffset = arguments->chunkStart - 1;

    double pih = 0.0;
    double fpisin = 0.0;

    int term_iteration = options->term_iteration;

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

    if (rank < lastRank)
    {
        MPI_Send(arguments->Matrix[m2][chunkSize - 1], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }

    if (rank > 0)
    {
        MPI_Recv(arguments->Matrix[m2][0], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(arguments->Matrix[m2][1], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    }

    if (rank < lastRank)
    {
        MPI_Recv(arguments->Matrix[m2][chunkSize], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    while (term_iteration > 0)
    {
        double** Matrix_Out = arguments->Matrix[m1];
        double** Matrix_In  = arguments->Matrix[m2];

        maxresiduum = 0;

        #pragma omp parallel for private(i, j, star, residuum) reduction(max: maxresiduum)
        /* over all rows */
        for (i = chunkSize - 1; i > 0; i--)
        {
            double fpisin_i = 0.0;

            if (options->inf_func == FUNC_FPISIN)
            {
                fpisin_i = fpisin * sin(pih * ((double)i + rowOffset));
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

        MPI_Allreduce(&maxresiduum, &recvmaxresiduum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        maxresiduum = recvmaxresiduum;

        results->stat_iteration++;
        results->stat_precision = maxresiduum;

        if (rank < lastRank)
        {
            MPI_Send(Matrix_Out[chunkSize - 1], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }

        if (rank > 0)
        {
            MPI_Recv(Matrix_Out[0], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(Matrix_Out[1], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }

        if (rank < lastRank)
        {
            MPI_Recv(Matrix_Out[chunkSize], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

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

static
void
displayMatrixSingle (struct calculation_arguments* arguments, struct calculation_results* results, struct options* options)
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

static
int
ownsLine(uint64_t rank, int size, int method, uint64_t from, uint64_t to, uint64_t line)
{
    if (method == METH_JACOBI)
    {
        return line >= from && line <= to;
    }
    else
    {
        return (line % size) == rank;
    }
}

/**
 * rank and size are the MPI rank and size, respectively.
 * from and to denote the global(!) range of lines that this process is responsible for.
 *
 * Example with 9 matrix lines and 4 processes:
 * - rank 0 is responsible for 1-2, rank 1 for 3-4, rank 2 for 5-6 and rank 3 for 7.
 *   Lines 0 and 8 are not included because they are not calculated.
 * - Each process stores two halo lines in its matrix (except for ranks 0 and 3 that only store one).
 * - For instance: Rank 2 has four lines 0-3 but only calculates 1-2 because 0 and 3 are halo lines for other processes. It is responsible for (global) lines 5-6.
 */
static
void
DisplayMatrixMPI (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options, int rank, int size, int from, int to)
{
    int const elements = 8 * options->interlines + 9;

    int x, y;
    double** Matrix = arguments->Matrix[results->m];
    MPI_Status status;

    /* first line belongs to rank 0 */
    if (rank == 0)
    {
        from--;
    }

    /* last line belongs to rank size - 1 */
    if (rank + 1 == size)
    {
        to++;
    }

    if (rank == 0)
    {
        printf("Matrix:\n");
    }

    for (y = 0; y < 9; y++)
    {
        int line = y * (options->interlines + 1);

        if (rank == 0)
        {
            /* check whether this line belongs to rank 0 */
            if (!ownsLine(rank, size, options->method, from, to, line))
            {
                /* use the tag to receive the lines in the correct order
                * the line is stored in Matrix[0], because we do not need it anymore */
                MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            if (ownsLine(rank, size, options->method, from, to, line))
            {
                /* if the line belongs to this process, send it to rank 0
                * (line - from + 1) is used to calculate the correct local address */
                MPI_Send(Matrix[lineToIndex(arguments, options, line)], elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
            }
        }

        if (rank == 0)
        {
            for (x = 0; x < 9; x++)
            {
                int col = x * (options->interlines + 1);

                if (ownsLine(rank, size, options->method, from, to, line))
                {
                    /* this line belongs to rank 0 */
                    printf("%7.4f", Matrix[lineToIndex(arguments, options, line)][col]);
                }
                else
                {
                    /* this line belongs to another rank and was received above */
                    printf("%7.4f", Matrix[0][col]);
                }
            }

            printf("\n");
        }
    }

    fflush(stdout);
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
    if (arguments->nprocs > 1)
    {
        DisplayMatrixMPI(arguments, results, options, arguments->rank, arguments->nprocs, arguments->chunkStart, arguments->chunkEnd);
    }
    else
    {
        displayMatrixSingle(arguments, results, options);
    }
    
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

    MPI_Init(&argc, &argv);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // world_rank = 1;
    // world_size = 2;

    askParams(&options, argc, argv, world_rank);
    MPI_Barrier(MPI_COMM_WORLD);
    initVariables(&arguments, &results, &options, world_rank, world_size);

    printf("Rank: %d, Size: %d, N: %ld, Interlines: %ld. Threads: %ld\n", world_rank, world_size, arguments.N, options.interlines, options.number);
    allocateMatrices(&arguments, &options);
    initMatrices(&arguments, &options);

    gettimeofday(&start_time, NULL);
    if (options.method == METH_JACOBI)
    {
        if (world_size == 1)
        {
            calculate_omp(&arguments, &results, &options);
        }
        else
        {
            calculate_mpi_jacobi(&arguments, &results, &options);
        }
    }
    else
    {
        if (world_size == 1)
        {
            calculate(&arguments, &results, &options);
        }
        else
        {
            calculate_mpi_gseidel(&arguments, &results, &options);
        }
    }
    gettimeofday(&comp_time, NULL);

    if (arguments.rank == 0)
    {
        displayStatistics(&arguments, &results, &options);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    displayMatrix(&arguments, &results, &options);

    freeMatrices(&arguments);

    MPI_Finalize();

    return 0;
}
