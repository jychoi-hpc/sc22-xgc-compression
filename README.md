# XGC compression workflow

This is to demostrate XGC data compression workflow.

```
           adios        adios
           write        read
  xgc_proxy -> xgc.f0.bp -> xgc_fdata_reader
(N processes)                (M processes)
```

## Command line example

```
jsrun -n $((8*48)) -c7 -g1 -r6 -brs xgc_proxy d3d_coarse_v2 48 400 500 10 60
jsrun -n $((8*2)) -c7 -g1 -r6 -brs xgc_fdata_reader
```
Note: 
* Each process of xgc_proxy (writer) writes a chunk of (1, nvp, nnodes/np_per_plane, nmu) shape
* Each process of xgc_fdata_reader (reader) reads a chunk of (nphi, nvp, nnodes/total_mpi_processes, nmu) shape
* For MGARD, we reorder dimension: (nphi, nvp, nnodes, nmu) => (nphi, nnodes, nvp, nmu)