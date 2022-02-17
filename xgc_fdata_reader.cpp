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
#define GET3D(X, d0, d1, d2, i, j, k) X[(d1 * d2) * i + d2 * j + k]
#define GET4D(X, d0, d1, d2, d3, i, j, k, l) X[(d1 * d2 * d3) * i + (d2 * d3) * j + d3 * k + l]

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << std::endl;
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

    if (argc < 1)
    {
        if (rank == 0)
            show_usage(argv[0]);
        return 1;
    }

    long unsigned int nphi = 0; // number of planes (will be set after reading)
    int np_per_plane = size;    // number of PEs per plane

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
        adios2::StepStatus status = reader.BeginStep();
        if (status != adios2::StepStatus::OK)
        {
            if (rank == 0)
                printf("%d: No more data. Stop\n", rank);
            break;
        }

        if (rank == 0)
            printf("%d: Reading xgc.f0.bp step: %d\n", rank, i);

        // Step #1: Read XGC data from xgc_proxy
        // Each process will read approximately nphi x nnodes/np_per_plane
        auto var_i_f = io.InquireVariable<double>("i_f");
        nphi = var_i_f.Shape()[0];
        assert(("Wrong number of MPI processes.", size == np_per_plane));
        long unsigned int nvp = var_i_f.Shape()[1];
        long unsigned int nnodes = var_i_f.Shape()[2];
        long unsigned int nmu = var_i_f.Shape()[3];

        long unsigned int l_nnodes = nnodes / np_per_plane;
        long unsigned int l_offset = rank * l_nnodes;
        if ((rank % np_per_plane) == (np_per_plane - 1))
            l_nnodes = l_nnodes + nnodes % np_per_plane;
        // printf("%d: nphi, l_offset, l_nnodes:\t%d\t%d\t%d\n", rank, nphi, l_offset, l_nnodes);
        var_i_f.SetSelection({{0, 0, l_offset, 0}, {nphi, nvp, l_nnodes, nmu}});

        std::vector<double> i_f;
        reader.Get<double>(var_i_f, i_f);
        reader.EndStep();

        /*
        // checking values
        for (int i = 0; i < nvp; i++)
            for (int j = 0; j < l_nnodes; j++)
                for (int k = 0; k < nmu; k++)
                    printf("%d: local i_f(\t%d,\t%d,\t%d\t) =\t%f\n", rank, i, j, k, GET3D(i_f, nvp, l_nnodes, nmu, i,
        j, k)); break; break; break;
        */

        // Step #2: do computation (compression and decompression, QoIs, etc)
        // Step #2a: chaning dimension order (nphi,nvp,nnodes,nmu) => (nphi,nnodes,nvp,nmu)
        std::vector<double> i_g(i_f.size());
        for (int i = 0; i < nphi; i++)
        {
            for (int j = 0; j < nvp; j++)
            {
                for (int k = 0; k < l_nnodes; k++)
                {
                    for (int l = 0; l < nmu; l++)
                    {
                        GET4D(i_g, nphi, l_nnodes, nvp, nmu, i, k, j, l) =
                            GET4D(i_f, nphi, nvp, l_nnodes, nmu, i, j, k, l);
                    }
                }
            }
        }

        // Step #3: Write something
        if (rank == 0)
            printf("%d: Writing: out.bp\n", rank);
        static bool first = true;
        if (first)
        {
            wio = ad.DeclareIO("writer");
            wio.DefineVariable<double>("i_f", {nphi, nvp, nnodes, nmu}, {0, 0, l_offset, 0},
                                       {nphi, nvp, l_nnodes, nmu});
            wio.DefineVariable<double>("i_g", {nphi, nnodes, nvp, nmu}, {0, l_offset, 0, 0},
                                       {nphi, l_nnodes, nvp, nmu});
            writer = wio.Open("out.bp", adios2::Mode::Write, comm);
            first = false;
        }

        writer.BeginStep();
        auto var_f = wio.InquireVariable<double>("i_f");
        auto var_g = wio.InquireVariable<double>("i_g");
        writer.Put<double>(var_f, i_f.data());
        writer.Put<double>(var_g, i_g.data());
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