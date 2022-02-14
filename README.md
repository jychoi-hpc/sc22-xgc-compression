# XGC compression proxy

This is to demostrate XGC data compression workflow.

```
           adios        adios
           write        read
  xgc_proxy -> xgc.f0.bp -> xgc_fdata_reader
(N processes)                (M processes)
```

