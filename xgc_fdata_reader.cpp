#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <assert.h>

#include "adios2.h"
#include "mpi.h"

#define GET2D(X, d0, d1, i, j) X[d1 * i + j]
#define GET3D(X, d0, d1, d2, i, j, k) X[(d1 + d2) * i + d2 * j + k]

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " NP_PER_PLANE" << std::endl;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0)
        printf("rank,size: %d %d\n", rank, size);

    if (argc < 2)
    {
        if (rank == 0)
            show_usage(argv[0]);
        return 1;
    }

    int np_per_plane = atoi(argv[1]); // number of PEs per plane

    if (rank == 0)
    {
        printf("np_per_plane: %d\n", np_per_plane);
    }
    MPI_Barrier(comm);

    long unsigned int nphi = 0;                   // number of planes (will be set after reading)
    long unsigned int iphi = rank / np_per_plane; // plane index
    int plane_rank = rank % np_per_plane;         // rank in plane
    // printf("%d: iphi, plane_rank:\t%d\t%d\n", rank, iphi, plane_rank);
    MPI_Barrier(comm);

    adios2::ADIOS ad(comm);
    adios2::IO io;
    adios2::Engine reader;

    adios2::IO wio;
    adios2::Engine writer;

    io = ad.DeclareIO("reader");

    int i = 0;
    reader = io.Open("xgc.f0.bp", adios2::Mode::Read, comm);
    while (true)
    {
        ++i;
        if (rank == 0)
            printf("%d: Reading xgc.f0.bp step: %d\n", rank, i);
        adios2::StepStatus status = reader.BeginStep();
        if (status != adios2::StepStatus::OK)
        {
            if (rank == 0)
                printf("%d: No more data. Stop\n", rank);
            break;
        }

        // Step #1: Read XGC data from xgc_proxy
        auto var_i_f = io.InquireVariable<double>("i_f");
        nphi = var_i_f.Shape()[0];
        assert(("Wrong number of MPI processes.", size == nphi * np_per_plane));
        long unsigned int nvp = var_i_f.Shape()[1];
        long unsigned int nnodes = var_i_f.Shape()[2];
        long unsigned int nmu = var_i_f.Shape()[3];

        long unsigned int l_nnodes = nnodes / np_per_plane;
        long unsigned int l_offset = plane_rank * l_nnodes;
        if ((plane_rank % np_per_plane) == (np_per_plane - 1))
            l_nnodes = l_nnodes + nnodes % np_per_plane;
        // printf("%d: iphi, l_offset, l_nnodes:\t%d\t%d\t%d\n", rank, iphi, l_offset, l_nnodes);
        var_i_f.SetSelection({{iphi, 0, l_offset, 0}, {1, nvp, l_nnodes, nmu}});

        std::vector<double> i_f;
        reader.Get<double>(var_i_f, i_f);
        reader.EndStep();

        /*
        for (int i = 0; i < nvp; i++)
            for (int j = 0; j < l_nnodes; j++)
                for (int k = 0; k < nmu; k++)
                    printf("%d: local i_f(\t%d,\t%d,\t%d\t) =\t%f\n", rank, i, j, k, GET3D(i_f, nvp, l_nnodes, nmu, i,
        j, k)); break; break; break;
        */

        // Step #2: do computation (compression and decompression, QoIs, etc)

        // Step #3: Write something
        if (rank == 0)
            printf("%d: Writing: out.bp\n", rank);
        static bool first = true;
        if (first)
        {
            wio = ad.DeclareIO("writer");
            wio.DefineVariable<double>("i_f", {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                       {1, nvp, l_nnodes, nmu});
            writer = wio.Open("out.bp", adios2::Mode::Write, comm);
            first = false;
        }

        writer.BeginStep();
        auto var = wio.InquireVariable<double>("i_f");
        writer.Put<double>(var, i_f.data());
        writer.EndStep();

        if (rank == 0)
            printf("%d: Finished step %d\n", rank, i);
    }

    reader.Close();
    writer.Close();
    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}