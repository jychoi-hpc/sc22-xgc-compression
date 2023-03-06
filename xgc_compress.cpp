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
#include <gptl.h>
#include <gptlmpi.h>
#include <papi.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

#define GET2D(X, d0, d1, i, j) X[d1 * i + j]
#define GET3D(X, d0, d1, d2, i, j, k) X[(d1 + d2) * i + d2 * j + k]
#define GET4D(X, d0, d1, d2, d3, i, j, k, l) X[(d1 + d2 + d3) * i + (d2 + d3) * j + d3 * k + l]

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " expdir method" << std::endl;
}

int main(int argc, char *argv[])
{
    int rank, comm_size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    if (rank == 0)
        printf("rank,comm_size: %d %d\n", rank, comm_size);

    std::string expdir = "./";
    std::string compname = "null";
    int maxstep = 0;
    int user_nnodes = 1000;

    // Optional arguments
    po::options_description desc("Allowed options");    
    desc.add_options()
        ("help,h", "produce help message")
        ("expdir,w", po::value(&expdir), "XGC directory")
        ("compname,c", po::value(&compname), "compression method")
        ("maxstep,s", po::value<int>(&maxstep)->default_value(1000), "max steps")
        ("nnodes,n", po::value<int>(&user_nnodes)->default_value(1000), "user nnodes");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    if (rank == 0)
        printf("expdir: %s\n", expdir.data());
    
    int ret;
    int code;
    //ret = GPTLsetoption (GPTL_IPC, 1);     // Count instructions per cycle
    //ret = GPTLsetoption (PAPI_TOT_INS, 1); // Print total instructions
    //ret = GPTLevent_name_to_code("example:::EXAMPLE_CONSTANT", &code);
    //ret = GPTLevent_name_to_code("nvml:::Tesla_V100-SXM2-16GB:device_0:gpu_utilization", &code);
    //ret = GPTLsetoption (code, 1);
    GPTLinitialize();
    MPI_Barrier(comm);

    long unsigned int nphi = 0;            // number of planes (will be set after reading)
    long unsigned int nplane_per_rank = 0; // number of planes per rank
    long unsigned int iphi = 0;            // plane index
    int plane_rank = 0;                    // rank in plane
    int np_per_plane = 0;

    adios2::ADIOS ad("adios2cfg.xml", comm);
    adios2::IO io = ad.DeclareIO("f0");
    adios2::IO wio = ad.DeclareIO("f0compress");
    adios2::Engine writer;

    char filename[50];
    long unsigned int nvp = 0;
    long unsigned int nnodes = 0;
    long unsigned int nmu = 0;
    long unsigned int l_nnodes = 0;
    long unsigned int l_offset = 0;
    static bool first = true;

    // Use yaml to pass parameters to Adios operator
    std::map<std::string, std::string> params;
    try
    {
        YAML::Node root = YAML::LoadFile("params.yaml");
        if (rank == 0)
            std::cout << "Operation parameters:" << std::endl;
        for (YAML::const_iterator it = root.begin(); it != root.end(); ++it)
        {
            if (rank == 0)
                std::cout << " " << it->first.as<std::string>() << ": " << it->second.as<std::string>() << std::endl;
            params.insert({it->first.as<std::string>(), it->second.as<std::string>()});
        }
    }
    catch (...)
    {
        // do nothing
    }

    sprintf(filename, "%s/restart_dir/xgc.f0.bp.000.bp", expdir.data());
    if (rank == 0)
        printf("%d: Reading filename: %s\n", rank, filename);
    adios2::Engine reader = io.Open(filename, adios2::Mode::Read, comm);

    for (int i = 0; i < maxstep; i += 1)
    {
        // Step #1: Read XGC data in step
        adios2::StepStatus read_status = reader.BeginStep();
        if (read_status != adios2::StepStatus::OK)
        {
            break;
        }

        int istep = reader.CurrentStep();

        adios2::Variable<double> var_i_f = io.InquireVariable<double>("i_f");
        adios2::Variable<double> var_e_f = io.InquireVariable<double>("e_f");

        if (first)
        {
            nphi = var_i_f.Shape()[0];        // number of planes
            np_per_plane = comm_size / nphi;  // number of PEs per plane
            if (np_per_plane < 1) np_per_plane = 1; // for debugging
            nplane_per_rank = 1;              // number of planes per rank
            iphi = rank / np_per_plane;       // plane index
            plane_rank = rank % np_per_plane; // rank in plane
            // assert(("Wrong number of MPI processes.", comm_size == nphi * np_per_plane));

            printf("%d: iphi, plane_rank:\t%d\t%d\n", rank, iphi, plane_rank);
            MPI_Barrier(comm);

            // Assume shape is not changing
            nvp = var_i_f.Shape()[1];
            nnodes = var_i_f.Shape()[2];
            nmu = var_i_f.Shape()[3];

            l_nnodes = nnodes / np_per_plane;
            l_offset = plane_rank * l_nnodes;
            if ((plane_rank % np_per_plane) == (np_per_plane - 1))
                l_nnodes = l_nnodes + nnodes % np_per_plane;

            int decomp_num = 0;
            if (params.find("decomp") != params.end())
                decomp_num = atoi(params["decomp"].c_str());
            if (decomp_num == 1 or decomp_num >= 5)
            {                           //  == 0) {
                nplane_per_rank = nphi; // number of planes per rank
                iphi = 0;               // plane index
                l_nnodes = nnodes / comm_size;
                l_offset = rank * l_nnodes;
                if (rank == comm_size - 1)
                {
                    l_nnodes += nnodes % comm_size;
                }
            }
            if (comm_size == 1) l_nnodes =  user_nnodes; // for debugging
            if (comm_size == 1) nplane_per_rank = 1; // for debugging
            printf("%d: Selection: (%d %d %d %d) (%d %d %d %d)\n", rank, iphi, 0, l_offset, 0, nplane_per_rank, nvp, l_nnodes, nmu);
        }
        var_i_f.SetSelection({{iphi, 0, l_offset, 0}, {nplane_per_rank, nvp, l_nnodes, nmu}});
        var_e_f.SetSelection({{iphi, 0, l_offset, 0}, {nplane_per_rank, nvp, l_nnodes, nmu}});

        if (rank == 0)
            printf("%d: Reading step: %d\n", rank, istep);
        std::vector<double> i_f;
        std::vector<double> e_f;
        GPTLstart("adios_read");
        reader.Get<double>(var_i_f, i_f);
        reader.Get<double>(var_e_f, e_f);
        // End of adios2 read step
        reader.EndStep();
        GPTLstop("adios_read");

        // Step #2: Simulate computation
        // int interval = 10;
        // for (int k = sleep_sec; k > 0; k = k - interval)
        // {
        //     if (rank == 0)
        //         printf("%d: Computation: %d seconds left.\n", rank, k);
        //     std::this_thread::sleep_for(std::chrono::seconds(MIN(k, interval)));
        // }

        // Step #3: Write f-data
        if (rank == 0)
            printf("%d: Writing: xgc.f0_%d.bp\n", rank, np_per_plane);
        char output_fname[128];
        char meshfile[128];
        sprintf(output_fname, "xgc.f0_%d.bp", np_per_plane);
        if (first)
        {
            auto var_i_f = wio.DefineVariable<double>("i_f", {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                                      {nplane_per_rank, nvp, l_nnodes, nmu});
            auto var_e_f = wio.DefineVariable<double>("e_f", {nphi, nvp, nnodes, nmu}, {iphi, 0, l_offset, 0},
                                                      {nplane_per_rank, nvp, l_nnodes, nmu});
            // make params
            sprintf(meshfile, "%s/xgc.f0.mesh.bp", expdir.data());
            // std::map<std::string, std::string> params = {{"tolerance", accu},
            //                                              {"mode", "ABS"},
            //                                              {"s", "0"},
            //                                              {"meshfile", meshfile},
            //                                              {"compression_method",
            //                                              "3"},
            //                                              {"pq", "0"},
            //                                              {"prec", precision},
            //                                              {"latent_dim", "4"},
            //                                              {"train", train_yes},
            //                                              {"species", argname},
            //                                              {"use_ddp", use_ddp},
            //                                              {"use_pretrain", use_pre},
            //                                              {"nepoch", epochs},
            //                                              {"ae_thresh", ae_error},
            //                                              {"pqbits", pqbits},
            //                                              {"lr", lr},
            //                                              {"leb", lbound},
            //                                              {"ueb", ubound},
            //                                              {"decomp", decomp}};
            std::map<std::string, std::string> extra = {{"meshfile", meshfile}};
            params.insert(extra.begin(), extra.end());

            // add operator
            params["species"] = "ion";
            var_i_f.AddOperation(compname, params);
            params["species"] = "electron";
            var_e_f.AddOperation(compname, params);
            writer = wio.Open(output_fname, adios2::Mode::Write, comm);
            first = false;
        }

        // Debugging: skip the first step
        /*
        if (i == 0) {
            if (rank == 0)
                printf("%d: Skip: %d\n", rank, i);
            GPTLreset();
            continue;
        }
        */

        writer.BeginStep();
        auto var_i = wio.InquireVariable<double>("i_f");
        auto var_e = wio.InquireVariable<double>("e_f");
        GPTLstart("adios_write");
        writer.Put<double>(var_i, i_f.data());
        writer.Put<double>(var_e, e_f.data());
        writer.EndStep();
        GPTLstop("adios_write");

        char fname[80];
        if (rank == 0) {
            sprintf(fname, "xgc_compress-timing-p%d-%d.txt", rank, i);
            GPTLpr_file(fname);
        }
        sprintf(fname, "xgc_compress-timing-sumary-%d.txt", i);
        GPTLpr_summary_file(MPI_COMM_WORLD, fname);

        if (rank == 0)
            printf("%d: Finished step %d\n", rank, istep);
        // reset at each iteration
        GPTLreset();
    }

    printf("%d: All done.\n", rank);
    reader.Close();
    writer.Close();
    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}
