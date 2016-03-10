prog := vec
prog_objs := vec.o

CC := clang
CFLAGS := -Wall -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
LDFLAGS := 

.PHONY: all clean

all: $(prog)

$(prog): $(prog_objs)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

vec.o: vec.c

clean:
	@rm -rf *.o $(prog)
