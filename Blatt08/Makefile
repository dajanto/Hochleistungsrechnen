# Common definitions
CC = mpicc

# Compiler flags, paths and gibraries
gFLAGS = -std=c11 -pedantic -Wall -Wextra -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff partdiff-par partdiff-par-hybrid
OBJS = partdiff.o askparams.o
MPI_OBJS = partdiff-par.o askparams.o
OMPI_OBJS = partdiff-par-hybrid.o askparams.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff-par: $(MPI_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(MPI_OBJS) $(LIBS)

partdiff-par-hybrid: $(OMPI_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OMPI_OBJS) $(LIBS)

# Rule to create *.o from *.c
%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(MPI_OBJS)
	$(RM) $(OMPI_OBJS)
	$(RM) $(TGTS)
