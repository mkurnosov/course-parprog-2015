prog := datasharing
prog_objs := datasharing.o

CC := gcc
CFLAGS := -std=c99 -g -fopenmp 
LDFLAGS := -lm -fopenmp

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

datasharing.o: datasharing.c

clean:
	@rm -rf *.o $(prog)
