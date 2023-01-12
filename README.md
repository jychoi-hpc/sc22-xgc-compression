# XGC compression workflow

This is to demostrate XGC data compression workflow.

```
           adios        adios
           write        read
  xgc_writer -> xgc.f0.bp -> xgc_compression
(N processes)                (M processes)
```

## Command line example

```
jsrun -n $((8*48)) -c7 -g1 -r6 -brs xgc_proxy /dir/to/xgc 48 400 500 10 60
jsrun -n $((8*2)) -c7 -g1 -r6 -brs xgc_compression /dir/to/xgc
```
Note: 
* Each process of xgc_proxy (writer) writes a chunk of (1, nvp, nnodes/np_per_plane, nmu) shape
* Each process of xgc_fdata_reader (reader) reads a chunk of (nphi, nvp, nnodes/total_mpi_processes, nmu) shape
* Use params.yaml for MGARDPlus parameters