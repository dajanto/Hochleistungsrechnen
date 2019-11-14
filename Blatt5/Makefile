# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c11 -pedantic -Wall -Wextra -O3 -ggdb -pthread
LFLAGS = $(CFLAGS)
LIBS   = -lm

TGTS = partdiff partdiff-posix partdiff-posix-keepalive
OBJS = partdiff.o askparams.o
P_OBJS = partdiff-posix.o askparams.o
P_K_OBJS = partdiff-posix-keepalive.o askparams.o

# Targets ...
all: $(TGTS)

partdiff: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff-posix: $(P_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(P_OBJS) $(LIBS)

partdiff-posix-keepalive: $(P_K_OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(P_K_OBJS) $(LIBS)

partdiff-posix.o: partdiff-posix.c Makefile

partdiff-posix-keepalive.o: partdiff-posix-keepalive.c Makefile

partdiff.o: partdiff.c Makefile

askparams.o: askparams.c Makefile

# Rule to create *.o from *.c
%.o: %.c
	$(CC) -c $(CFLAGS) $*.c

clean:
	$(RM) $(OBJS)
	$(RM) $(TGTS)
