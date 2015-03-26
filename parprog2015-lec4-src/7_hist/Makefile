histeq := hist
histeq_objs := hist.o

CC := gcc
CFLAGS := -g -std=c99 -O2 -fopenmp 
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(histeq)

$(histeq): $(histeq_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

hist.o: hist.c

clean:
	@rm -rf *.o $(histeq)
