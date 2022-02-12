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

    if (argc < 7)
    {
        show_usage(argv[0]);
        return 1;
    }

    std::string expdir = argv[1];
    int nphi = atoi(argv[2]);  // number of planes
    int np = atoi(argv[3]);    // number of PEs per plane
    int istep = atoi(argv[4]); // start step index
    int nstep = atoi(argv[5]); // end step index
    int inc = atoi(argv[6]);   // inc

    if (rank == 0)
    {
        printf("expdir: %s\n", expdir.data());
        printf("nphi: %d\n", nphi);
        printf("np: %d\n", np);
        printf("istep: %d\n", istep);
        printf("nstep: %d\n", nstep);
        printf("inc: %d\n", inc);
    }
    MPI_Barrier(comm);

    int iphi = rank / np;       // plane index
    int plane_rank = rank % np; // rank in plane
    printf("%d: iphi, plane_rank: %d %d\n", rank, iphi, plane_rank);
    MPI_Barrier(comm);

    assert(size == nphi * np);

    adios2::ADIOS ad(comm);
    adios2::IO io;
    adios2::Engine reader;

    adios2::IO wio;
    adios2::Engine writer;

    io = ad.DeclareIO("reader");

    char filename[50];
    for (int i = istep; i < istep + nstep; i += inc)
    {
        sprintf(filename, "%s/restart_dir/xgc.f0.%05d.bp", expdir.data(), i);
        printf("%d: filename: %s\n", rank, filename);

        reader = io.Open(filename, adios2::Mode::Read, comm);
        auto var_i_f = io.InquireVariable<double>("i_f");

        assert(nphi == var_i_f.Shape()[0]);
        int nvp = var_i_f.Shape()[1];
        int nnodes = var_i_f.Shape()[2];
        int nmu = var_i_f.Shape()[3];

        int l_nnodes = nnodes / np;
        int l_offset = plane_rank * l_nnodes;
        if ((plane_rank % np) == (np - 1))
            l_nnodes = l_nnodes + nnodes % np;
        // printf("%d: nnodes, l_offset, l_nnodes: %d %d\n", rank, nnodes, l_offset, l_nnodes);

        printf("%d: iphi, l_offset, l_nnodes: %d %d %d\n", rank, iphi, l_offset, l_nnodes);
        var_i_f.SetSelection({{iphi, 0, l_offset, 0}, {1, nvp, l_nnodes, nmu}});

        std::vector<double> i_f;
        reader.Get<double>(var_i_f, i_f);

        reader.Close();

        /*
        for (int i = 0; i < nvp; i++)
            for (int j = 0; j < l_nnodes; j++)
                for (int k = 0; k < nmu; k++)
                    printf("%d: local i_f(\t%d,\t%d,\t%d\t) =\t%f\n", rank, i, j, k, GET3D(i_f, nvp, l_nnodes, nmu, i,
        j, k)); break; break; break;
        */

        // Writing back for debugg
        static bool first = true;
        if (first)
        {
            wio = ad.DeclareIO("writer");
            wio.DefineVariable<double>("i_f", {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                       {1, nvp, l_nnodes, nmu});

            writer = wio.Open("dump.bp", adios2::Mode::Write, comm);

            first = false;
        }

        writer.BeginStep();
        auto var = wio.InquireVariable<double>("i_f");
        writer.Put<double>(var, i_f.data());
        writer.EndStep();
    }

    writer.Close();
    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}