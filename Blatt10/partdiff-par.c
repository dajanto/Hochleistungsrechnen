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
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <malloc.h>
#include <sys/time.h>

#include "partdiff.h"
#include <mpi.h>
#include <omp.h>

struct calculation_results
{
    uint64_t m;
    uint64_t stat_iteration; /* number of current iteration                    */
    double stat_precision;   /* actual precision of all slaves in iteration    */
};

struct calculation_arguments
{
    uint64_t N;            /* number of spaces between lines (lines=N+1)     */
    uint64_t num_matrices; /* number of matrices                             */
    double h;              /* length of a space between two lines            */
    double ***Matrix;      /* index matrix used for addressing M             */
    double *M;             /* two matrices with real values                  */
    uint64_t chunkSize;
    uint64_t chunkStart;
    uint64_t chunkEnd;
    int rank;
    int nprocs;
    void (*calculateFunction)(const struct calculation_arguments *, struct calculation_results *, struct options const *);
    double **first_columns_cache;
    double **halo_lines_cache;
};

/* ************************************************************************ */
/* Global variables                                                         */
/* ************************************************************************ */

/* time measurement variables */
struct timeval start_time; /* time when program started                      */
struct timeval comp_time;  /* time when calculation completed                */

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static void *
allocateMemory(size_t size)
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
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
static void *
allocateMemoryEmpty(size_t count, size_t size)
{
    void *p;

    if ((p = calloc(count, size)) == NULL)
    {
        printf("Speicherprobleme! (%" PRIu64 " Bytes angefordert)\n", size * count);
        exit(1);
    }

    return p;
}

/* ************************************************************************ */
/* freeMatrices: frees memory for matrices                                  */
/* ************************************************************************ */
static void
freeMatrices(struct calculation_arguments *arguments)
{
    uint64_t i;
    for (i = 0; i < arguments->num_matrices; i++)
    {
        free(arguments->Matrix[i]);
    }

    free(arguments->Matrix);
    free(arguments->M);

    if (arguments->halo_lines_cache)
    {
        uint64_t cache_size = (arguments->nprocs - arguments->rank) * 2;
        for (i = 0; i < cache_size; i++)
        {
            free(arguments->halo_lines_cache[i]);
        }
        free(arguments->halo_lines_cache);
    }

    if (arguments->first_columns_cache)
    {
        uint64_t first_columns_size = (arguments->nprocs - arguments->rank);
        for (i = 0; i < first_columns_size; i++)
        {
            free(arguments->first_columns_cache[i]);
        }
        free(arguments->first_columns_cache);
    }
}

/* ************************************************************************ */
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static void
allocateMatrices(struct calculation_arguments *arguments, struct options const *options)
{
    uint64_t i, j;

    uint64_t const N = arguments->N;
    uint64_t const chunkSize = arguments->chunkSize;
    uint64_t const chunkLines = chunkSize + 1;

    arguments->M = allocateMemory(arguments->num_matrices * (chunkLines) * (N + 1) * sizeof(double));
    arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double **));

    for (i = 0; i < arguments->num_matrices; i++)
    {
        arguments->Matrix[i] = allocateMemory((chunkLines) * sizeof(double *));

        for (j = 0; j <= chunkSize; j++)
        {
            arguments->Matrix[i][j] = arguments->M + (i * (chunkLines) * (N + 1)) + (j * (N + 1));
        }
    }

    double **halo_lines_cache = NULL;
    double **first_columns_cache = NULL;

    if (options->method == METH_GAUSS_SEIDEL && arguments->nprocs > 1)
    {
        uint64_t cache_size = (arguments->nprocs - arguments->rank) * 2;
        uint64_t first_columns_size = (arguments->nprocs - arguments->rank);

        halo_lines_cache = allocateMemory(cache_size * sizeof(double *));
        first_columns_cache = allocateMemory(first_columns_size * sizeof(double *));

        for (i = 0; i < cache_size; i++)
        {
            halo_lines_cache[i] = allocateMemoryEmpty(N + 1, sizeof(double));
        }

        for (i = 0; i < first_columns_size; i++)
        {
            first_columns_cache[i] = allocateMemoryEmpty(chunkSize + 1, sizeof(double));
        }
    }
    arguments->first_columns_cache = first_columns_cache;
    arguments->halo_lines_cache = halo_lines_cache;
}

static double *
get_line(const struct calculation_arguments *arguments, struct calculation_results *results, uint64_t line)
{
    return arguments->Matrix[results->m][line - arguments->chunkStart + 1];
}

static int
ownsLine(uint64_t from, uint64_t to, uint64_t line)
{
    return line >= from && line <= to;
}

static uint64_t
indexToLine(const struct calculation_arguments *arguments, uint64_t index)
{
    // printf("i: %5ld, offset - 1: %5ld, result: %5ld\n", index, arguments->chunkStart - 1, index + arguments->chunkStart - 1);
    return index + arguments->chunkStart - 1;
}

/* ************************************************************************ */
/* initMatrices: Initialize matrix/matrices and some global variables       */
/* ************************************************************************ */
static void
initMatrices(struct calculation_arguments *arguments, struct options const *options)
{
    uint64_t g, i, j; /*  local variables for loops   */

    uint64_t const N = arguments->N;
    uint64_t const chunkSize = arguments->chunkSize;
    double const h = arguments->h;
    uint64_t const from = arguments->chunkStart;
    uint64_t const to = arguments->chunkEnd;
    double ***Matrix = arguments->Matrix;

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
                uint64_t line = indexToLine(arguments, i);

                Matrix[g][i][0] = 1.0 - (h * line);
                Matrix[g][i][N] = h * line;
                // printf("Rank %2d, Line: %2ld, g: %ld, i: %2ld, 0: %7.4f, N: %7.4f\n", arguments->rank, line, g, i, 1.0 - (h * line), (h * line));
            }
            // Matrix[g][N][0] = 0.0;
            // Matrix[g][0][N] = 0.0;
        }
        int ownsFirstLine = ownsLine(from, to, 1);
        int ownsLastLine = ownsLine(from, to, N - 1);

        // printf("Rank %2d owns Line 0: %d, Line N: %d\n", rank, ownsFirstLine, ownsLastLine);
        for (g = 0; g < arguments->num_matrices; g++)
        {
            for (i = 0; i <= N; i++)
            {
                if (ownsFirstLine)
                {
                    Matrix[g][0][i] = 1.0 - (h * i);
                }

                if (ownsLastLine)
                {
                    Matrix[g][chunkSize][i] = h * i;
                }
            }
        }
    }
}

static char *
formatChunk(double *chunk, uint64_t size)
{
    uint64_t maxLength = size * 7 + 1;
    char *buffer = calloc(maxLength, sizeof(char));
    int index = 0;
    for (uint64_t i = 0; i < size; i++)
    {
        int written = snprintf(&buffer[index], maxLength - index, "%7.4f", chunk[i]);
        if (written < 0)
        {
            printf("Could not append item to buffer\n");
            exit(1);
        }
        index += written;
    }
    return buffer;
}

static char *
formatChunkUI(uint64_t *chunk, uint64_t size)
{
    uint64_t maxLength = size * 7 + 1;
    char *buffer = calloc(maxLength, sizeof(char));
    int index = 0;
    for (uint64_t i = 0; i < size; i++)
    {
        int written = snprintf(&buffer[index], maxLength - index, "%ld ", chunk[i]);
        if (written < 0)
        {
            printf("Could not append item to buffer\n");
            exit(1);
        }
        index += written;
    }
    return buffer;
}

static void
printChunk(double *chunk, uint64_t size, uint64_t rank, int iteration, uint64_t index, uint64_t line, char *before, char *after)
{
    char *chunkChars = formatChunk(chunk, size);
    printf("%sRank %2ld Iteration %2d, Row %2ld(%2ld): %s%s\n", before ? before : "", rank, iteration, index, line, chunkChars, after ? after : "");
    free(chunkChars);
}

static void
printChunkUI(uint64_t *chunk, uint64_t size, uint64_t rank, int iteration, uint64_t index, uint64_t line, char *before, char *after)
{
    char *chunkChars = formatChunkUI(chunk, size);
    printf("%sRank %2ld Iteration %2d, Row %2ld(%2ld): %s%s\n", before ? before : "", rank, iteration, index, line, chunkChars, after ? after : "");
    free(chunkChars);
}

static void
printRows(double **Matrix, uint64_t row, int printPrior, int printNext, struct calculation_arguments *arguments, uint64_t term_iteration)
{
    if (printPrior)
    {
        printChunk(Matrix[row - 1], arguments->N + 1, arguments->rank, term_iteration, row - 1, indexToLine(arguments, row) - 1, "", "");
    }
    printChunk(Matrix[row], arguments->N + 1, arguments->rank, term_iteration, row, indexToLine(arguments, row), "", "");
    if (printNext)
    {
        printChunk(Matrix[row + 1], arguments->N + 1, arguments->rank, term_iteration, row + 1, indexToLine(arguments, row) + 1, "", "");
    }
    fflush(stdout);
}

/* ************************************************************************ */
/* calculate: solves the equation sequentially for either jacobi or gauß    */
/* ************************************************************************ */
static void
calculate(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
    uint64_t i, j;      /* local variables for loops */
    int m1, m2;         /* used as indices for old and new matrices */
    double star;        /* four times center value minus 4 neigh.b values */
    double residuum;    /* residuum of current iteration */
    double maxresiduum; /* maximum residuum value of a slave in iteration */

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
        double **Matrix_Out = arguments->Matrix[m1];
        double **Matrix_In = arguments->Matrix[m2];

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
                star = 0.25 * (Matrix_In[i - 1][j] + Matrix_In[i][j - 1] + Matrix_In[i][j + 1] + Matrix_In[i + 1][j]);

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
            // printChunk(Matrix_In[i], N + 1, 0, term_iteration, i, i);
            // fflush(stdout);
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

static void
reachIteration(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options, int finalIteration, double **halo_lines, int cache_i, double **first_columns, int column_i)
{
    uint64_t i, j; /* local variables for loops */
    double star;   /* four times center value minus 4 neigh.b values */
    uint64_t current_line_index;
    int current_iteration = results->stat_iteration;
    uint64_t chunkSize = arguments->chunkSize;
    uint64_t N = arguments->N;
    double const h = arguments->h;
    uint64_t rowOffset = arguments->chunkStart - 1;
    int cache_size = (arguments->nprocs - arguments->rank) * 2;
    int columns_size = cache_size / 2;
    double **Matrix = arguments->Matrix[0];

    double pih = 0.0;
    double fpisin = 0.0;

    if (options->inf_func == FUNC_FPISIN)
    {
        pih = PI * h;
        fpisin = TWO_PI_SQUARE * h * h;
    }

    if (current_iteration == finalIteration)
    {
        return;
    }
    else if (current_iteration < finalIteration)
    {
        printf("Rank %d Error: No Process should lag behind in Iterations, Expected greater than %d, Got %d\n", arguments->rank, finalIteration, current_iteration);
        exit(1);
    }

    while (current_iteration > finalIteration)
    {
        /* the C operator % is the remainder operator, not modulo (differences in handling negative numbers) -1 mod 2 = 1 , -1 % 2 = -1 */
        cache_i = (cache_i + cache_size - 2) % cache_size;
        column_i = (column_i + columns_size - 1) % columns_size;

        /* over all rows in reverse */
        for (i = chunkSize - 1; i > 0; i--)
        {
            current_line_index = i + rowOffset;
            double *current_line = Matrix[i];
            double *previous_line = Matrix[i - 1];
            double *next_line = Matrix[i + 1];
            if (i == 1 && arguments->rank > 0)
            {
                previous_line = halo_lines[cache_i];
            }
            else if ((chunkSize - 1) == i)
            {
                next_line = halo_lines[cache_i + 1];
            }
            double fpisin_i = 0.0;

            if (options->inf_func == FUNC_FPISIN)
            {
                fpisin_i = fpisin * sin(pih * (double)current_line_index);
            }
            /* over all columns in reverse */
            for (j = N - 2; j > 0; j--)
            {
                star = (4 * current_line[j]) - previous_line[j] - current_line[j - 1] - next_line[j];

                if (options->inf_func == FUNC_FPISIN)
                {
                    star -= fpisin_i * sin(pih * (double)j);
                }

                current_line[j + 1] = star;
            }
            current_line[1] = first_columns[column_i][i];
            // printRows(Matrix_In, i, i == 1, i == (chunkSize - 1), arguments, term_iteration);
        }

        current_iteration--;
    }
    results->stat_iteration = current_iteration;
    // printf("Rank %d: Final Iteration should be %d, Currrent: %ld\n",arguments->rank, finalIteration, results->stat_iteration);
}

/* ************************************************************************ */
/* calculate: solves the equation of gauß seidel in parallel with mpi only  */
/* ************************************************************************ */
static void
calculate_mpi_gseidel_block(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
    uint64_t i, j;          /* local variables for loops */
    int m1, m2;             /* used as indices for old and new matrices */
    double star;            /* four times center value minus 4 neigh.b values */
    double residuum;        /* residuum of current iteration */
    double maxresiduum = 0; /* maximum residuum value of a slave in iteration */

    uint64_t const N = arguments->N;
    double const h = arguments->h;
    int term_iteration = options->term_iteration;

    uint64_t chunkSize = arguments->chunkSize;
    uint64_t rowOffset = arguments->chunkStart - 1;
    int nprocs = arguments->nprocs;
    int rank = arguments->rank;
    int lastRank = nprocs - 1;
    uint64_t current_line;
    int previous_rank = rank - 1;
    int next_rank = rank + 1;
    int cache_i = 0;
    int column_i = 0;

    int finalIteration = 0;
    int current_iteration = 0;
    int check_end_iteration = nprocs - rank - 1;

    double pih = 0.0;
    double fpisin = 0.0;

    /* as this algorithm is exclusive for gauß seidel, no check is needed */
    m1 = 0;
    m2 = 0;

    if (options->inf_func == FUNC_FPISIN)
    {
        pih = PI * h;
        fpisin = 0.25 * TWO_PI_SQUARE * h * h;
    }

    uint64_t cache_size = (nprocs - rank) * 2;
    uint64_t first_columns_size = (nprocs - rank);
    double **halo_lines_cache = arguments->halo_lines_cache;
    double **first_columns_cache = arguments->first_columns_cache;

    const int SEND_ROW_DOWN_TAG = 1;
    const int SEND_ROW_UP_TAG = 2;
    const int SEND_RESIDUUM_DOWN_TAG = 3;
    const int SEND_FINISHED_UP_TAG = 4;

    while (term_iteration > 0)
    {
        double **Matrix_Out = arguments->Matrix[m1];
        double **Matrix_In = arguments->Matrix[m2];

        if (rank > 0)
        {
            if (options->termination == TERM_PREC && current_iteration >= check_end_iteration)
            {
                // the last rank calculates always the smallest iteration, so it needs to signal when it should finish
                if (rank == lastRank)
                {
                    finalIteration = maxresiduum < options->term_precision ? current_iteration : 0;
                }
                //send the value of finalIteration to the previous neighbour, always zero if it should continue
                MPI_Send(&finalIteration, 1, MPI_INT, previous_rank, SEND_FINISHED_UP_TAG, MPI_COMM_WORLD);
            }
            // send the row with old values to the previous neighbor
            MPI_Send(Matrix_In[1], N + 1, MPI_DOUBLE, previous_rank, SEND_ROW_UP_TAG, MPI_COMM_WORLD);

            // receive the first row with current values for this iteration
            MPI_Recv(Matrix_In[0], N + 1, MPI_DOUBLE, previous_rank, SEND_ROW_DOWN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (options->termination == TERM_PREC)
            {
                // get the maxresiduum for this iteration from previous rank
                MPI_Recv(&maxresiduum, 1, MPI_DOUBLE, previous_rank, SEND_RESIDUUM_DOWN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            if (cache_size)
            {
                memcpy(halo_lines_cache[cache_i], Matrix_In[0], (N + 1) * sizeof(double));
            }
        }

        if (finalIteration)
        {
            break;
        }

        if (rank < lastRank)
        {
            if (options->termination == TERM_PREC && current_iteration >= (check_end_iteration - 1))
            {
                MPI_Recv(&finalIteration, 1, MPI_INT, next_rank, SEND_FINISHED_UP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            MPI_Recv(Matrix_In[chunkSize], N + 1, MPI_DOUBLE, next_rank, SEND_ROW_UP_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Rank %2d, Iteration %d, Receiving Last Halo Line\n", rank, current_iteration);

            if (cache_size)
            {
                memcpy(halo_lines_cache[cache_i + 1], Matrix_In[chunkSize], (N + 1) * sizeof(double));
            }
        }

        if (rank == 0 || options->termination == TERM_ITER)
        {
            maxresiduum = 0;
        }

        /* over all rows */
        for (i = 1; i < chunkSize; i++)
        {
            current_line = i + rowOffset;

            double fpisin_i = 0.0;

            if (options->inf_func == FUNC_FPISIN)
            {
                fpisin_i = fpisin * sin(pih * (double)current_line);
            }
            /* over all columns */
            for (j = 1; j < N; j++)
            {
                star = 0.25 * (Matrix_In[i - 1][j] + Matrix_In[i][j - 1] + Matrix_In[i][j + 1] + Matrix_In[i + 1][j]);

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
            // printRows(Matrix_In, i, i == 1, i == (chunkSize - 1), arguments, term_iteration);
        }

        for (i = 0; i < chunkSize; i++)
        {
            first_columns_cache[column_i][i] = Matrix_In[i][1];
        }

        if (rank < lastRank)
        {
            MPI_Send(Matrix_In[chunkSize - 1], N + 1, MPI_DOUBLE, next_rank, SEND_ROW_DOWN_TAG, MPI_COMM_WORLD);

            if (options->termination == TERM_PREC)
            {
                MPI_Send(&maxresiduum, 1, MPI_DOUBLE, next_rank, SEND_RESIDUUM_DOWN_TAG, MPI_COMM_WORLD);
            }
        }
        current_iteration++;
        results->stat_precision = maxresiduum;

        if (cache_size)
        {
            cache_i = (cache_i + 2) % cache_size;
            column_i = (column_i + 1) % first_columns_size;
        }
        if (options->termination == TERM_ITER)
        {
            term_iteration--;
        }
    }
    results->stat_iteration = current_iteration;
    results->m = 0;

    if (options->termination == TERM_PREC)
    {
        MPI_Bcast(&results->stat_precision, 1, MPI_DOUBLE, lastRank, MPI_COMM_WORLD);
        /* Wrong? backwards calculation, wrong values, especially the lower the rank */
        /* reachIteration(arguments, results, options, finalIteration, halo_lines_cache, cache_i, first_columns_cache, column_i); */
    }
    else
    {
        MPI_Reduce(&maxresiduum, &results->stat_precision, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
}

/* ************************************************************************ */
/* calculate: solves the equation with jacobi in paralel with open mp only  */
/* ************************************************************************ */
static void
calculate_omp(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
    uint64_t i, j;      /* local variables for loops */
    int m1, m2;         /* used as indices for old and new matrices */
    double star;        /* four times center value minus 4 neigh.b values */
    double residuum;    /* residuum of current iteration */
    double maxresiduum; /* maximum residuum value of a slave in iteration */

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
        double **Matrix_Out = arguments->Matrix[m1];
        double **Matrix_In = arguments->Matrix[m2];

        maxresiduum = 0;

#pragma omp parallel for private(i, j, star, residuum) reduction(max \
                                                                 : maxresiduum)
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
                star = 0.25 * (Matrix_In[i - 1][j] + Matrix_In[i][j - 1] + Matrix_In[i][j + 1] + Matrix_In[i + 1][j]);

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
static void
calculate_mpi_jacobi(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options)
{
    uint64_t i, j;          /* local variables for loops */
    int m1, m2;             /* used as indices for old and new matrices */
    double star;            /* four times center value minus 4 neigh.b values */
    double residuum;        /* residuum of current iteration */
    double maxresiduum;     /* maximum residuum value of a slave in iteration */
    double recvmaxresiduum; /* tmp holder for maximum residuum value of global in iteration */

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
        double **Matrix_Out = arguments->Matrix[m1];
        double **Matrix_In = arguments->Matrix[m2];

        maxresiduum = 0;

#pragma omp parallel for private(i, j, star, residuum) reduction(max \
                                                                 : maxresiduum)
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
                star = 0.25 * (Matrix_In[i - 1][j] + Matrix_In[i][j - 1] + Matrix_In[i][j + 1] + Matrix_In[i + 1][j]);

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
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static void
initVariables(struct calculation_arguments *arguments, struct calculation_results *results, struct options const *options, uint64_t rank, uint64_t nprocs)
{
    arguments->N = (options->interlines * 8) + 9 - 1;
    arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
    arguments->h = 1.0 / arguments->N;
    results->m = 0;
    results->stat_iteration = 0;
    results->stat_precision = 0;

    if (arguments->N < 50 || options->term_iteration < 50)
    {
        nprocs = 1;
    }

    uint64_t total_lines = arguments->N - 1;
    const uint64_t min_lines = 5;

    if (min_lines > (total_lines / nprocs))
    {
        nprocs = total_lines / min_lines;
    }

    uint64_t chunkSize = total_lines / nprocs;
    uint64_t chunkRest = total_lines % nprocs;

    if (chunkRest && chunkRest > rank)
    {
        chunkSize++;
    }

    uint64_t start = 1;
    uint64_t end;

    if (nprocs > 1 && rank < nprocs)
    {
        if (rank > 0)
        {
            MPI_Recv(&start, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        end = start + chunkSize - 1;

        if (rank < (nprocs - 1))
        {
            uint64_t nextStart = end + 1;
            MPI_Send(&nextStart, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        end = start + chunkSize - 1;
    }

    if (options->method == METH_JACOBI)
    {
        if (nprocs == 1)
        {
            arguments->calculateFunction = calculate_omp;
            chunkSize = arguments->N;
        }
        else
        {
            arguments->calculateFunction = calculate_mpi_jacobi;
            chunkSize++;
        }
    }
    else
    {
        if (nprocs == 1)
        {
            arguments->calculateFunction = calculate;
            chunkSize = arguments->N;
        }
        else
        {
            arguments->calculateFunction = calculate_mpi_gseidel_block;
            chunkSize++;
        }
    }

    if (rank >= nprocs)
    {
        arguments->calculateFunction = NULL;
    }

    arguments->chunkSize = chunkSize;
    arguments->chunkStart = start;
    arguments->chunkEnd = end;
    arguments->rank = rank;
    arguments->nprocs = nprocs;
}

/* ************************************************************************ */
/*  displayStatistics: displays some statistics about the calculation       */
/* ************************************************************************ */
static void
displayStatistics(struct calculation_arguments const *arguments, struct calculation_results const *results, struct options const *options)
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
    printf("Interlines:         %" PRIu64 "\n", options->interlines);
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

static void
displayMatrixSingle(struct calculation_arguments *arguments, struct calculation_results *results, struct options *options)
{
    int x, y;

    double **Matrix = arguments->Matrix[results->m];

    int const interlines = options->interlines;

    printf("Matrix:\n");
#ifdef ALL
    int end = arguments->N + 1;
#else
    int end = 9;
#endif
    for (y = 0; y < end; y++)
    {
        for (x = 0; x < end; x++)
        {
#ifdef ALL
            printf("%7.4f", Matrix[y][x]);
#else
            printf("%7.4f", Matrix[y * (interlines + 1)][x * (interlines + 1)]);
#endif
        }

        printf("\n");
    }

    fflush(stdout);
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
static void
DisplayMatrixMPI(struct calculation_arguments const *arguments, struct calculation_results *results, struct options const *options, int rank, int size, int from, int to)
{
    int const elements = 8 * options->interlines + 9;

    int x, y;
    double **Matrix = arguments->Matrix[results->m];
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

    // printf("Rank %d owns: %s\n", rank, formatChunkUI(arguments->indexTable, arguments->chunkSize));

    if (rank == 0)
    {
        printf("Matrix:\n");
    }
#ifdef ALL
    int end = arguments->N + 1;
#else
    int end = 9;
#endif
    for (y = 0; y < end; y++)
    {
#ifdef ALL
        uint64_t line = y;
#else
        uint64_t line = y * (options->interlines + 1);
#endif
        if (rank == 0)
        {
            /* check whether this line belongs to rank 0 */
            if (!ownsLine(from, to, line))
            {
                // printf("Waiting for line %ld\n", line);
                /* use the tag to receive the lines in the correct order
                * the line is stored in Matrix[0], because we do not need it anymore */
                MPI_Recv(Matrix[0], elements, MPI_DOUBLE, MPI_ANY_SOURCE, 42 + y, MPI_COMM_WORLD, &status);
            }
        }
        else
        {
            if (ownsLine(from, to, line))
            {
                double *line_array = get_line(arguments, results, line);
                /* if the line belongs to this process, send it to rank 0
                * (line - from + 1) is used to calculate the correct local address */
                MPI_Send(line_array, elements, MPI_DOUBLE, 0, 42 + y, MPI_COMM_WORLD);
            }
            else
            {
                // printf("Rank %d does not own line %ld\n", rank, line);
            }
        }

        if (rank == 0)
        {
            for (x = 0; x < end; x++)
            {
#ifdef ALL
                int col = x;
#else
                int col = x * (options->interlines + 1);
#endif

                if (ownsLine(from, to, line))
                {
                    double *line_array = get_line(arguments, results, line);
                    /* this line belongs to rank 0 */
                    printf("%7.4f", line_array[col]);
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
static void
displayMatrix(struct calculation_arguments *arguments, struct calculation_results *results, struct options *options)
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
int main(int argc, char **argv)
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

    // printf(
    //     "Rank: %d, Size: %d, N: %ld, Interlines: %ld. Threads: %ld, Chunksize: %ld, ChunkStart: %ld, ChunkEnd: %ld\n",
    //     world_rank,
    //     world_size,
    //     arguments.N,
    //     options.interlines,
    //     options.number,
    //     arguments.chunkSize,
    //     arguments.chunkStart,
    //     arguments.chunkEnd
    // );
    if (arguments.calculateFunction)
    {
        allocateMatrices(&arguments, &options);
        initMatrices(&arguments, &options);

        gettimeofday(&start_time, NULL);
        arguments.calculateFunction(&arguments, &results, &options);
        gettimeofday(&comp_time, NULL);

        if (arguments.rank == 0)
        {
            displayStatistics(&arguments, &results, &options);
        }
        displayMatrix(&arguments, &results, &options);
        freeMatrices(&arguments);
    }
    MPI_Finalize();

    return 0;
}
