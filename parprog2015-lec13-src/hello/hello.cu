/*
 * hello.cu:
 *
 *
 */

#include <stdio.h>

__global__ void mykernel() 
{
    
}

int main()
{
    mykernel<<<1,1>>>();
    printf("Hello, CUDA World!\n");
    return 0;
}

