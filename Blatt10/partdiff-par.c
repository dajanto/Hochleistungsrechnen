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
    uint64_t  m;
    uint64_t  stat_iteration; /* number of current iteration                    */
    double    stat_precision; /* actual precision of all slaves in iteration    */
};

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
    void  (*calculateFunction)(const struct calculation_arguments*, struct calculation_results*, struct options const*);
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
/* allocateMatrices: allocates memory for matrices                          */
/* ************************************************************************ */
static
void
allocateMatrices (struct calculation_arguments* arguments)
{
    uint64_t i, j;

    uint64_t const N = arguments->N;
    uint64_t const chunkSize = arguments->chunkSize;
    uint64_t const chunkLines = chunkSize + 1;

    arguments->M = allocateMemory(arguments->num_matrices * (chunkLines) * (N + 1) * sizeof(double));
    arguments->Matrix = allocateMemory(arguments->num_matrices * sizeof(double**));

    for (i = 0; i < arguments->num_matrices; i++)
    {
        arguments->Matrix[i] = allocateMemory((chunkLines) * sizeof(double*));

        for (j = 0; j <= chunkSize; j++)
        {
            arguments->Matrix[i][j] = arguments->M + (i * (chunkLines) * (N + 1)) + (j * (N + 1));
        }
    }
}

static
double*
get_line(const struct calculation_arguments* arguments, struct calculation_results* results, uint64_t line)
{
    return arguments->Matrix[results->m][line - arguments->chunkStart + 1];
}

static
int
ownsLine(uint64_t from, uint64_t to, uint64_t line)
{
    return line >= from && line <= to;
}

static
uint64_t
indexToLine(const struct calculation_arguments* arguments, uint64_t index)
{
    // printf("i: %5ld, offset - 1: %5ld, result: %5ld\n", index, arguments->chunkStart - 1, index + arguments->chunkStart - 1);
    return index + arguments->chunkStart - 1;
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
    uint64_t const from = arguments->chunkStart;
    uint64_t const to = arguments->chunkEnd;
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

/* ************************************************************************ */
/* calculate: solves the equation of gauß seidel in parallel with mpi only  */
/* ************************************************************************ */
static
void
calculate_mpi_gseidel_wave (struct calculation_arguments const* arguments, struct calculation_results* results, struct options const* options)
{
  uint64_t i, j;                                   /* local variables for loops */
  int m1, m2;                                 /* used as indices for old and new matrices */
  double star;                                /* four times center value minus 4 neigh.b values */
  double residuum;                            /* residuum of current iteration */
  double maxresiduum;                         /* maximum residuum value of a slave in iteration */
  int flag;                                   //flag die signalisiert. dass der letzte Prozess die gewünschte Genaugkeit erreicht hat.

  uint64_t const N = arguments->N;
  double const h = arguments->h;
  int chunkSize = arguments->chunkSize;
  int rank = arguments->rank;
  int lastRank = arguments->nprocs - 1;
  // int rowOffset = arguments->chunkStart - 1;  //brauche ich wahrscheinlich nicht

  double pih = 0.0;
  double fpisin = 0.0;

  int term_iteration = options->term_iteration;

  /* initialize m1 and m2 depending on algorithm */

      m1 = 0;
      m2 = 0;


  if (options->inf_func == FUNC_FPISIN)
  {
      pih = PI * h;
      fpisin = 0.25 * TWO_PI_SQUARE * h * h;
  }

//Senden der ersten Zeile an den Vorgänger + Empfangen der 1. Zeile des Nachfolgers, vor dem Rechnen
    if (rank > 0)
    {
        MPI_Send(arguments->Matrix[m2][1], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    }

    if (rank < lastRank)
    {
        MPI_Recv(arguments->Matrix[m2][chunkSize], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

  MPI_Request status;
  MPI_Irecv(&flag, 1, MPI_INT, lastRank, 1, MPI_COMM_WORLD, &status);

  while (term_iteration > 0)
  {
      double** Matrix_Out = arguments->Matrix[m1];
      double** Matrix_In  = arguments->Matrix[m2];

    //  if(rank == 0)

      maxresiduum = 0;

      //Empfangen der letzten Zeile des Vorgängers
      if (rank > 0)
      {
        MPI_Recv(Matrix_Out[0], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        maxresiduum = Matrix_Out[0][0];   //Das maxresiduum des Vorgängerblockes wird in der übertragenen Zeile in der oberen linken Ecke, welche für die Berechnungen irrelevant ist weitergegeben

        if(maxresiduum < 0)
        {
          flag = 1;
          maxresiduum = - maxresiduum;
        }
      }

      /* over all rows */
      for (i = 1; i < (uint64_t) chunkSize; i++)  //unsigned casten um die Warning zu unterdrücken
      {
          double fpisin_i = 0.0;

          if (options->inf_func == FUNC_FPISIN)
          {
              fpisin_i = fpisin * sin(pih * (double) indexToLine(arguments, i));
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

          //Verschicken der ersten Zeile an den Vorgänger (damit dieser weiter machen kann) nach der ersten Iteration
          if (i == 1 && rank > 0 && !flag)
          {
            MPI_Send(Matrix_Out[1], N + 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
          }
      }

      results->stat_iteration++;
      results->stat_precision = maxresiduum;

      if(rank == 0)
      {
        MPI_Test(&status, &flag, MPI_STATUS_IGNORE);
      }

      if(flag)
      {
          maxresiduum = - maxresiduum;
      }

      //Das maxresiduum in die letzte Zeile in die Ecke schreiben, Wert ersetzen und zurückersetzen
      double buffer;
      buffer = Matrix_Out[chunkSize - 1][0];

      if(rank < lastRank)
      {
        Matrix_Out[chunkSize - 1][0] = maxresiduum;
      }

      // Verschicken der letzten Zeile an den Nachfolger; das bisherige maxresiduum der Iteration ist darin enthalten
      if (rank < lastRank)
      {
          MPI_Send(Matrix_Out[chunkSize - 1], N + 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
      }

      //zurückersetzen des Wertes in Matrix_Out
      if(rank < lastRank)
      {
        Matrix_Out[chunkSize - 1][0] = buffer;
      }

      // Warten auf die erste Zeile des Nachfolgers
      if (rank < lastRank && !flag)
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
          if (maxresiduum < 0)          // wenn ein Prozess die Nachricht bekommt,dass er fertig ist, soll er aufhören
          {
              term_iteration = 0;
          }

          if ((rank == lastRank) && (0 < maxresiduum) && (maxresiduum < options->term_precision))
          {
              int indicator = 1;
              MPI_Send(&indicator, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);   //benachrichtigen des 0. Prozesses, das die gewünschte Genaugkeit erreicht wurde; Tag 1
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
/* initVariables: Initializes some global variables                         */
/* ************************************************************************ */
static
void
initVariables (struct calculation_arguments* arguments, struct calculation_results* results, struct options const* options, uint64_t rank, uint64_t nprocs)
{
    arguments->N = (options->interlines * 8) + 9 - 1;
    arguments->num_matrices = (options->method == METH_JACOBI) ? 2 : 1;
    arguments->h = 1.0 / arguments->N;
    results->m = 0;
    results->stat_iteration = 0;
    results->stat_precision = 0;

    /*
      Führe die sequentielle Variante aus, falls sich die parallele nicht lohnt.
    */
    // if (arguments->N < 50 || options->term_iteration < 50)
    //     {
    //         nprocs = 1;
    //     }


        uint64_t total_lines = arguments->N - 1;
        const uint64_t min_lines = 1;                 //Setze mindeste Anzahl an Zeilen die ein Prozess bearbeiten soll auf 1 (zur sinnvollen Abarbeitung)

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
                arguments->calculateFunction = calculate_mpi_gseidel_wave;
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

    // printf("Rank %d owns: %s\n", rank, formatChunkUI(arguments->indexTable, arguments->chunkSize));

    if (rank == 0)
    {
        printf("Matrix:\n");
    }

    for (y = 0; y < 9; y++)
    {
        uint64_t line = y * (options->interlines + 1);

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
            }else
            {
                // printf("Rank %d does not own line %ld\n", rank, line);
            }
        }

        if (rank == 0)
        {
            for (x = 0; x < 9; x++)
            {
                int col = x * (options->interlines + 1);

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
        allocateMatrices(&arguments);
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
