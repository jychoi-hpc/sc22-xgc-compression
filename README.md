# XGC compression workflow

This is to demostrate XGC data compression workflow.

```
          adios        adios
          write        read
        XGC -> xgc.f0.bp -> xgc_compress
(N processes)                (M processes)
```

## Command line example

```
cat << EOF > params.yaml
tolerance: 1e16
mode: ABS
s: 0
compression_method: 0
pq: 0
prec: single
latent_dim: 4
train: 0
use_ddp: 0
use_pretrain: 0
nepoch: 100
ae_thresh: 0.001
pqbits: 4
lr: 1e-5
leb: -1
ueb: -1
decomp: 0
EOF

jsrun -n192 -c7 -g1 -r6 -brs xgc_compression /dir/to/xgc mgardplus
```
Note: 
* Use params.yaml for MGARDPlus parameters