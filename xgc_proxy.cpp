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
#define GET4D(X, d0, d1, d2, d3, i, j, k, l) X[(d1 + d2 + d3) * i + (d2 + d3) * j + d3 * k + l]

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " EXPDIR NP_PER_PLANE BSTEP ESTEP INC SLEEP_SEC" << std::endl;
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

    if (argc < 7)
    {
        if (rank == 0)
            show_usage(argv[0]);
        return 1;
    }

    std::string expdir = argv[1];
    int np_per_plane = atoi(argv[2]); // number of PEs per plane
    int bstep = atoi(argv[3]);        // start step index
    int estep = atoi(argv[4]);        // end step index
    int inc = atoi(argv[5]);          // inc
    int sleep_sec = atoi(argv[6]);

    if (rank == 0)
    {
        printf("expdir: %s\n", expdir.data());
        printf("np_per_plane: %d\n", np_per_plane);
        printf("bstep: %d\n", bstep);
        printf("estep: %d\n", estep);
        printf("inc: %d\n", inc);
        printf("sleep_sec: %d\n", sleep_sec);
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

    char filename[50];
    for (int i = bstep; i < estep; i += inc)
    {
        // Step #1: Read original XGC data
        sprintf(filename, "%s/restart_dir/xgc.f0.%05d.bp", expdir.data(), i);
        if (rank == 0)
            printf("%d: Reading filename: %s\n", rank, filename);

        reader = io.Open(filename, adios2::Mode::Read, comm);
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
        reader.Close();

        /*
        for (int i = 0; i < nvp; i++)
            for (int j = 0; j < l_nnodes; j++)
                for (int k = 0; k < nmu; k++)
                    printf("%d: local i_f(\t%d,\t%d,\t%d\t) =\t%f\n", rank, i, j, k, GET3D(i_f, nvp, l_nnodes, nmu, i,
        j, k)); break; break; break;
        */

        // Step #2: Simulate computation
        int interval = 10;
        for (int k = sleep_sec; k > 0; k = k - interval)
        {
            if (rank == 0)
                printf("%d: Computation: %d seconds left.\n", rank, k);
            std::this_thread::sleep_for(std::chrono::seconds(MIN(k, interval)));
        }

        // Step #3: Write f-data
        if (rank == 0)
            printf("%d: Writing: xgc.f0.bp\n", rank);
        static bool first = true;
        if (first)
        {
            wio = ad.DeclareIO("writer");
            auto var = wio.DefineVariable<double>("i_f", {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                       {1, nvp, l_nnodes, nmu});
            // add operator
            var.AddOperation("mgardplus",{{"accuracy","0.01"}});
            writer = wio.Open("xgc.f0.bp", adios2::Mode::Write, comm);

            first = false;
        }

        writer.BeginStep();
        auto var = wio.InquireVariable<double>("i_f");
        writer.Put<double>(var, i_f.data());
        writer.EndStep();

        if (rank == 0)
            printf("%d: Finished step %d\n", rank, i);
    }

    writer.Close();
    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}