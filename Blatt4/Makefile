# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -O3 -ggdb -fopenmp
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff partdiff-openmp-zeilen partdiff-openmp-element partdiff-openmp-spalten
OBJS = partdiff.o askparams.o
ROW_OBJS = partdiff-openmp-zeilen.o askparams.o
ELEMENT_OBJS = partdiff-openmp-element.o askparams.o
COLUMN_OBJS = partdiff-openmp-spalten.o askparams.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff-openmp: $(ROW_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(ROW_OBJS) $(LIBS)

partdiff-openmp-zeilen: $(ROW_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(ROW_OBJS) $(LIBS)

partdiff-openmp-spalten: $(COLUMN_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(COLUMN_OBJS) $(LIBS)

partdiff-openmp-element: $(ELEMENT_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(ELEMENT_OBJS) $(LIBS)

partdiff.o: partdiff.c Makefile

askparams.o: askparams.c Makefile

partdiff-openmp-zeilen.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D ROWS

partdiff-openmp-spalten.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D COLUMNS

partdiff-openmp-element.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D ELEMENTS

# Rule to create *.o from *.c
%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
