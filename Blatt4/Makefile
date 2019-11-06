# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -O3 -ggdb -fopenmp
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff partdiff_p_rows partdiff_p_columns partdiff_p_elements
OBJS = partdiff.o askparams.o
ROW_OBJS = partdiff_p_rows.o askparams.o
ELEMENT_OBJS = partdiff_p_elements.o askparams.o
COLUMN_OBJS = partdiff_p_columns.o askparams.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff_p_rows: $(ROW_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(ROW_OBJS) $(LIBS) -D ROWS

partdiff_p_columns: $(COLUMN_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(COLUMN_OBJS) $(LIBS) -D COLUMNS

partdiff_p_elements: $(ELEMENT_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(ELEMENT_OBJS) $(LIBS) -D ELEMENTS

partdiff.o: partdiff.c Makefile

askparams.o: askparams.c Makefile

partdiff_p_rows.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D ROWS

partdiff_p_columns.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D COLUMNS

partdiff_p_elements.o: $(OBJS) Makefile
	$(CC) -o $@ -c $(CFLAGS) partdiff.c -D ELEMENTS

# Rule to create *.o from *.c
%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
