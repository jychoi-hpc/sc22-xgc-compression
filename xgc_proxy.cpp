#include <iostream>
#include <string>
#include <vector>

#include <assert.h>

#include "adios2.h"
#include "mpi.h"

#define GET2D(X, d0, d1, i, j) X[d1 * i + j]
#define GET3D(X, d0, d1, d2, i, j, k) X[(d1 + d2) * i + d2 * j + k]

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " EXPDIR NPHI NP ISTEP NSTEP INC" << std::endl;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    printf("rank,size: %d %d\n", rank, size);

    MPI_Finalize();
    return 0;
}