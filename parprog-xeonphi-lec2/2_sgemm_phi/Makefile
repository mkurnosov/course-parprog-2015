prog := sgemm
prog_objs := sgemm.o

CC := icc
CFLAGS := -Wall -g -std=c99 -O2 -fopenmp
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sgemm.o: sgemm.c

clean:
	@rm -rf *.o $(prog)
