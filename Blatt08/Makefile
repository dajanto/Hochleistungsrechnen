# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff
OBJS = partdiff.o askparams.o displaymatrix-mpi.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

# Rule to create *.o from *.c
%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
