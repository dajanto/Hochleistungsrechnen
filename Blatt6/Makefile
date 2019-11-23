# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = timempi timempi2
OBJS = timempi.o timempi2.o
# Targets ...
all: $(TGTS)

timempi: timempi.o Makefile
	$(CC) $(LFLAGS) -o $@ timempi.c $(LIBS)

timempi2: timempi2.o Makefile
	$(CC) $(LFLAGS) -o $@ timempi2.c $(LIBS)

# Rule to create *.o from *.c
%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
