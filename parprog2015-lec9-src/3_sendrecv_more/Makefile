prog := sendrecv
prog_objs := sendrecv.o

CC := mpicc
CFLAGS := -std=c99 -Wall -O2
LDFLAGS :=

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

sendrecv.o: sendrecv.c

clean:
	@rm -rf *.o $(prog)
