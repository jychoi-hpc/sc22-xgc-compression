#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <assert.h>

#include "adios2.h"
#include "mpi.h"

#include <yaml-cpp/yaml.h>

#define GET2D(X, d0, d1, i, j) X[d1 * i + j]
#define GET3D(X, d0, d1, d2, i, j, k) X[(d1 + d2) * i + d2 * j + k]
#define GET4D(X, d0, d1, d2, d3, i, j, k, l) X[(d1 + d2 + d3) * i + (d2 + d3) * j + d3 * k + l]

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " EXPDIR NP_PER_PLANE BSTEP ESTEP INC SLEEP_SEC [NNODES]" << std::endl;
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
    char *accu = argv[7];
    int ptype = atoi(argv[8]);
    char *precision = argv[9];
    char *train_yes = argv[10];
    char *use_pre = argv[11];
    char *use_ddp = argv[12];
    char *epochs = argv[13];
    char *ae_error = argv[14];
    char *pqbits = argv[15];
    char *lr = argv[16];
    char *decomp = argv[17];
    char *lbound = argv[18];
    char *ubound = argv[19];
    int user_nnodes = 0; // user defined nnodes (optional)
    if (argc > 20)
        user_nnodes = atoi(argv[20]);

    if (rank == 0)
    {
        printf("expdir: %s\n", expdir.data());
        printf("np_per_plane: %d\n", np_per_plane);
        printf("bstep: %d\n", bstep);
        printf("estep: %d\n", estep);
        printf("inc: %d\n", inc);
        printf("sleep_sec: %d\n", sleep_sec);
        printf("accuracy: %s\n", accu);
        printf("species: %d\n", ptype);
        printf("precision: %s\n", precision);
        printf("train: %s\n", train_yes);
        printf("use_pre: %s\n", use_pre);
        printf("use_ddp: %s\n", use_ddp);
        printf("epochs: %s\n", epochs);
        printf("ae_error: %s\n", ae_error);
        printf("pqbits: %s\n", pqbits);
        printf("learning rate: %s\n", lr);
        printf("decomposition: %s\n", decomp);
        printf("lbound: %s\n", lbound);
        printf("ubound: %s\n", ubound);
        printf("user_nnodes: %d\n", user_nnodes);
    }
    MPI_Barrier(comm);
    long unsigned int nphi = 0;                   // number of planes (will be set after reading)
    long unsigned int nplane_per_rank = 1;        // number of planes per rank
    long unsigned int iphi = rank / np_per_plane; // plane index
    int plane_rank = rank % np_per_plane;         // rank in plane
    // printf("%d: iphi, plane_rank:\t%d\t%d\n", rank, iphi, plane_rank);
    MPI_Barrier(comm);

    // Use yaml for extra parameters
    std::map<std::string, std::string> params2;
    try
    {
        YAML::Node config = YAML::LoadFile("config.yaml");
        if (rank == 0)
            std::cout << "Operation parameters:" << std::endl;
        for (YAML::const_iterator it = config.begin(); it != config.end(); ++it)
        {
            if (rank == 0)
                std::cout << " " << it->first.as<std::string>() << ": " << it->second.as<std::string>() << std::endl;
            params2.insert({it->first.as<std::string>(), it->second.as<std::string>()});
        }
    }
    catch (...)
    {
        // do nothing
    }

    adios2::ADIOS ad(comm);
    adios2::IO io;
    adios2::Engine reader;

    adios2::IO wio;
    adios2::Engine writer;

    const char *varname = ptype == 1 ? "i_f" : "e_f";
    const char *argname = ptype == 1 ? "ion" : "electron";
    char ioname[50];
    char filename[50];
    long unsigned int nvp = 0;
    long unsigned int nnodes = 0;
    long unsigned int nmu = 0;
    long unsigned int l_nnodes = 0;
    long unsigned int l_offset = 0;
    static bool first = true;
    for (int i = bstep; i < estep; i += inc)
    {
        // Step #1: Read original XGC data
        sprintf(filename, "%s/restart_dir/xgc.f0.%05d.bp", expdir.data(), i);
        if (rank == 0)
            printf("%d: Reading filename: %s\n", rank, filename);

        sprintf(ioname, "reader%d", i);
        io = ad.DeclareIO(ioname);
        reader = io.Open(filename, adios2::Mode::Read, comm);
        adios2::Variable<double> var_i_f = io.InquireVariable<double>(varname);

        nphi = var_i_f.Shape()[0];
        assert(("Wrong number of MPI processes.", size == nphi * np_per_plane));
        nvp = var_i_f.Shape()[1];
        nnodes = var_i_f.Shape()[2];
        nmu = var_i_f.Shape()[3];

        l_nnodes = nnodes / np_per_plane;
        l_offset = plane_rank * l_nnodes;
        if ((plane_rank % np_per_plane) == (np_per_plane - 1))
            l_nnodes = l_nnodes + nnodes % np_per_plane;

        // use user-defined nnodes if specified
        if (user_nnodes > 0)
        {
            l_nnodes = user_nnodes;
            l_offset = plane_rank * l_nnodes;
            nnodes = size * user_nnodes;
        }
        int decomp_num = atoi(decomp);
        if (decomp_num == 1 or decomp_num >= 5)
        {                           //  == 0) {
            nplane_per_rank = nphi; // number of planes per rank
            iphi = 0;               // plane index
            l_nnodes = nnodes / size;
            l_offset = rank * l_nnodes;
            if (rank == size - 1)
            {
                l_nnodes += nnodes % size;
            }
        }
        printf("%d: Selection: (%d %d %d %d) (%d %d %d %d)\n", rank, iphi, 0, l_offset, 0, nplane_per_rank, nvp,
               l_nnodes, nmu);
        var_i_f.SetSelection({{iphi, 0, l_offset, 0}, {nplane_per_rank, nvp, l_nnodes, nmu}});

        std::vector<double> i_f;
        reader.Get<double>(var_i_f, i_f);
        reader.Close();

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
            printf("%d: Writing: xgc.f0_%d.bp\n", rank, np_per_plane);
        char output_fname[50];
        sprintf(output_fname, "xgc.f0_%d.bp", np_per_plane);
        if (first)
        {
            wio = ad.DeclareIO("writer");
            auto var = wio.DefineVariable<double>(varname, {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                                  {nplane_per_rank, nvp, l_nnodes, nmu});
            // make params
            std::map<std::string, std::string> params = {{"tolerance", accu},
                                                         {"mode", "ABS"},
                                                         {"s", "0"},
                                                         {"meshfile", "exp-22013-ITER/xgc.f0.mesh.bp"},
                                                         {"compression_method", "3"},
                                                         {"pq", "0"},
                                                         {"prec", precision},
                                                         {"latent_dim", "4"},
                                                         {"train", train_yes},
                                                         {"species", argname},
                                                         {"use_ddp", use_ddp},
                                                         {"use_pretrain", use_pre},
                                                         {"nepoch", epochs},
                                                         {"ae_thresh", ae_error},
                                                         {"pqbits", pqbits},
                                                         {"lr", lr},
                                                         {"leb", lbound},
                                                         {"ueb", ubound},
                                                         {"decomp", decomp}};
            params.insert(params2.begin(), params2.end());

            // add operator
            // var.AddOperation("mgardplus", params);
            // var.AddOperation("mgardplus",{{"tolerance", accu}, {"mode", "ABS"}, {"s", "0"}, {"meshfile",
            // "d3d_coarse_small_v2/xgc.f0.mesh.bp"}, {"compression_method", "3"}, {"pq", "0"}, {"precision",
            // precision}, {"ae", "/gpfs/alpine/csc143/proj-shared/tania/sc22-xgc-compression/ae/my_ae.pt"},
            // {"latent_dim", "5"}, {"batch_size", "128"}, {"train", "1"}, {"species", argname},{"use_ddp",
            // use_ddp},{"use_pretrain", use_pre},{"nepoch", epochs},{"ae_thresh", ae_error},{"pqbits", pqbits}});
            writer = wio.Open(output_fname, adios2::Mode::Write, comm);
            first = false;
        }

        writer.BeginStep();
        auto var = wio.InquireVariable<double>(varname);
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
