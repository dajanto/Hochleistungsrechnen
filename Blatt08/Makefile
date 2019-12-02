# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff partdiff-mpi
OBJS = partdiff.o askparams.o
MPI_OBJS = partdiff-mpi.o askparams.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff-mpi: $(MPI_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

# Rule to create *.o from *.c
%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(MPI_OBJS)
	$(RM) $(TGTS)
