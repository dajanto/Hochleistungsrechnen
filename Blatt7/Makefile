# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = circle
OBJS = circle.o
# Targets ...
all: $(TGTS)

circle: circle.o Makefile
	$(CC) $(LFLAGS) -o $@ circle.c $(LIBS)

# Rule to create *.o from *.c
%.o: %.c Makefile
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
