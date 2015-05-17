#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

enum cmdtype {
    CMD_TERMINATE = 0,
    CMD_LOAD = 1,
    CMD_STORE = 2
};

enum {
    PACKSIZE_MAX = 256
};

struct cmd_load {
    int data1;
    double data2;
};

struct cmd_store {
    double data1;
    int data2;
    char data3[5];
};

char packedbuf[PACKSIZE_MAX];
MPI_Datatype cmdloadtype;
MPI_Datatype cmdstoretype;

void create_mpitypes()
{
    struct cmd_load load;
    int blocklens[2] = {1, 1};
    MPI_Aint displ[2];
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};

    struct cmd_store store;
    int store_blocklens[3] = {1, 1, 5};
    MPI_Aint store_displ[3];
    MPI_Datatype store_types[3] = {MPI_DOUBLE, MPI_INT, MPI_CHAR};

    displ[0] = (void *)&load.data1 - (void *)&load;
    displ[1] = (void *)&load.data2 - (void *)&load;
    MPI_Type_create_struct(2, blocklens, displ, types, &cmdloadtype);
    MPI_Type_commit(&cmdloadtype);

    store_displ[0] = (void *)&store.data1 - (void *)&store;
    store_displ[1] = (void *)&store.data2 - (void *)&store;
    store_displ[2] = (void *)&store.data3 - (void *)&store;
    MPI_Type_create_struct(3, store_blocklens, store_displ, store_types, &cmdstoretype);
    MPI_Type_commit(&cmdstoretype);
}

void free_mpitypes()
{
    MPI_Type_free(&cmdloadtype);
    MPI_Type_free(&cmdstoretype);
}

void run_handler()
{
    int cmd, pos, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for (int i = 0; i < 5; i++) {
        /* Send random command */        
        int cmd = (rand() > RAND_MAX / 2) ? CMD_LOAD : CMD_STORE;
        pos = 0;
        MPI_Pack(&cmd, 1, MPI_INT, packedbuf, PACKSIZE_MAX, &pos, MPI_COMM_WORLD);

        if (cmd == CMD_LOAD) {
            struct cmd_load cmdload;
            cmdload.data1 = 0xDEADBEAF;
            cmdload.data2 = 3.14;
            MPI_Pack(&cmdload, 1, cmdloadtype, packedbuf, PACKSIZE_MAX, &pos, MPI_COMM_WORLD);
            printf("Handler %d CMD_LOAD pos = %d\n", rank, pos);
            MPI_Send(packedbuf, pos, MPI_PACKED, 0, 0, MPI_COMM_WORLD);
            MPI_Status status;
            struct cmd_load response;
            MPI_Recv(&response, 1, cmdloadtype, 0, 0, MPI_COMM_WORLD, &status);
            printf("Handler %d CMD_LOAD data1 = %d\n", rank, response.data1);
        } else if (cmd == CMD_STORE) {
            struct cmd_store cmdstore;
            cmdstore.data1 = 0.27;
            cmdstore.data2 = 333;
            MPI_Pack(&cmdstore, 1, cmdstoretype, packedbuf, PACKSIZE_MAX, &pos, MPI_COMM_WORLD);
            printf("Handler %d CMD_STORE pos = %d\n", rank, pos);
            MPI_Send(packedbuf, pos, MPI_PACKED, 0, 0, MPI_COMM_WORLD);
        }
    }
    /* Send terminate command */
    cmd = CMD_TERMINATE;
    pos = 0;
    MPI_Pack(&cmd, 1, MPI_INT, packedbuf, PACKSIZE_MAX, &pos, MPI_COMM_WORLD);
    MPI_Send(packedbuf, pos, MPI_PACKED, 0, 0, MPI_COMM_WORLD);
}

void run_store()
{
    int commsize;   
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    int nhandlers = commsize - 1;
    
    while (nhandlers > 0) {
        MPI_Status status;
        MPI_Recv(packedbuf, PACKSIZE_MAX, MPI_PACKED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        printf("Storage recv %d bytes\n", count);
        
        /* Unpack command */
        int cmd, pos = 0;
        MPI_Unpack(packedbuf, PACKSIZE_MAX, &pos, &cmd, 1, MPI_INT, MPI_COMM_WORLD);

        if (cmd == CMD_TERMINATE) {
            nhandlers--;
            printf("Handler %d terminated\n", status.MPI_SOURCE);
        } else if (cmd == CMD_STORE) {
            struct cmd_store cmdstore;
            MPI_Unpack(packedbuf, PACKSIZE_MAX, &pos, &cmdstore, 1, cmdstoretype, MPI_COMM_WORLD);
            printf("Storage received CMD_STORE from %d (data %.6f)\n", status.MPI_SOURCE, cmdstore.data1);
        } else if (cmd == CMD_LOAD) {
            struct cmd_load cmdload;
            MPI_Unpack(packedbuf, PACKSIZE_MAX, &pos, &cmdload, 1, cmdloadtype, MPI_COMM_WORLD);
            printf("Storage received CMD_LOAD from %d (data 0x%x)\n", status.MPI_SOURCE, cmdload.data1);
            cmdload.data1 = 38;
            MPI_Send(&cmdload, 1, cmdloadtype, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char **argv)
{
    int rank, commsize;    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    create_mpitypes();

    // Handlers send load/store commands to the storage process
    if (rank > 0)
        run_handler();
    else
        run_store();

    free_mpitypes();    
    MPI_Finalize();
    return 0;
}
