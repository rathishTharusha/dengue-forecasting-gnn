# Exact reproduction workspace

Runs the **authors' own code** to obtain the **authors' own numbers**. Nothing
here reimplements a model, a metric, a training loop or a data pipeline — that is
what `crosscheck/` is for, and the two workspaces answer different questions:

| Workspace | Question |
|---|---|
| `crosscheck/` | Are our own implementations right, and are the papers' methods sound? |
| `reproduction/` | Does the authors' released code produce the numbers they published? |

**Read [`REPRODUCIBILITY_MATRIX.md`](REPRODUCIBILITY_MATRIX.md) first.** It records,
with evidence, exactly which published numbers can be reproduced and which
cannot. The headline: of Weng et al.'s 20 Table I rows, **5 are reproducible** —
but they are the five GNN rows, which are the paper's actual contribution.

---

## Kernels

| Kernel | Reproduces | Target |
|---|---|---|
| `weng-2024-exact-reproduction-stgat` | Weng et al., STGAT | MAE 25.38, RMSE 44.78 |
| `weng-2024-exact-reproduction-a3tgcn` | Weng et al., A3TGCN | MAE 34.03, RMSE 58.52 |
| `weng-2024-exact-reproduction-astgcn` | Weng et al., ASTGCN | MAE 33.68, RMSE 47.72 |
| `weng-2024-exact-reproduction-dcrnn` | Weng et al., DCRNN | MAE 45.98, RMSE 71.61 |
| `weng-2024-exact-reproduction-aagcn` | Weng et al., AAGCN | MAE 41.93, RMSE 55.83 |
| `seir-model-reproduction-gopalakrishnan` | Gopalakrishnan SEIR | 6 qualitative claims |
| `seir-sei-sensitivity-reproduction-phaijoo-gurung` | Phaijoo & Gurung Table 1 | 9 sensitivity indices |

Targets come from `Models/results.txt` in the authors' repository — their own raw
output, matching Table I to the decimal — not from the rounded printed table.

There is **no DengueGNN kernel**: the article publishes no code and does not
specify which OpenDengue subset it used, so there is nothing to run that would
constitute a reproduction.

---

## Order of operations

**1. Verify the stack locally** (~10 min, mostly downloading torch):

```bash
python reproduction/verify_local.py
```

Builds the pinned environment, clones the authors' repo at commit `45f1c08`, and
runs all five architectures for one epoch on one segment. The numbers it prints
are meaningless; the point is that their code executes. Confirmed working:

```
run_stgat    OK
run_a3tgcn   OK
run_astgcn   OK
run_dcrnn    OK
run_aagcn    OK
```

**2. Set up the Kaggle CLI** — see [`kaggle/README.md`](kaggle/README.md). You
must create your own API token; that step cannot be done for you.

```bash
python reproduction/kaggle/setup_kaggle.py --check
python reproduction/kaggle/setup_kaggle.py --init-metadata
```

**3. Push one kernel at a time.** Kaggle caps a session at 12 hours and each
architecture takes roughly 1–4 hours on CPU:

```bash
cd reproduction/kaggle/kernels/weng-2024-exact-reproduction-stgat
kaggle kernels push
kaggle kernels status <your-username>/weng-2024-exact-reproduction-stgat
kaggle kernels output <your-username>/weng-2024-exact-reproduction-stgat -p ../../../results/
```

**4. Log the result** in `docs/EXPERIMENT_LOG.md`, per the repo convention.

---

## Two forced deviations

The authors' `requirements.txt` cannot be honoured as written — it is broken in
two independent ways, both documented with the exact error in
[`REPRODUCIBILITY_MATRIX.md`](REPRODUCIBILITY_MATRIX.md):

1. `torch_geometric_temporal==0.54.0` declares `pandas<=1.3.5`, contradicting the
   file's own `pandas~=2.2.0` → PGT installed with `--no-deps`.
2. PGT 0.54.0 imports `torch_geometric.utils.to_dense_adj`, a module PyG removed
   in 2.4, so the pinned pair **cannot import** → `torch_geometric==2.4.0`, the
   newest version that works.

Everything else is installed at exactly the version the authors named. Both
deviations appear in every notebook's header and must be restated wherever these
numbers are cited.

---

## Layout

```
reproduction/
├── README.md                      this file
├── REPRODUCIBILITY_MATRIX.md      what can and cannot be reproduced, with evidence
├── verify_local.py                build the pinned stack and smoke-test it
├── _build/                        notebook cell sources — EDIT THESE
│   ├── gen_kernels.py
│   ├── cells_weng.py
│   └── cells_seir.py
├── kaggle/
│   ├── README.md                  CLI setup and push/status/output commands
│   ├── setup_kaggle.py            verify auth; fill your username into metadata
│   └── kernels/<slug>/            notebook + kernel-metadata.json, ready to push
├── notebooks/                     reading copies (git-ignored; kernels/ is source)
└── results/                       downloaded kernel outputs
```

Notebooks are **generated**, same convention as `notebooks/_build` and
`crosscheck/_build`:

```bash
cd reproduction/_build && python gen_kernels.py
```

Edit `cells_*.py`, never the `.ipynb` — regeneration overwrites them.
`tools/check_notebooks.py` validates every generated cell parses as Python.

---

## What a successful reproduction would and would not establish

It would confirm the authors ran what they said they ran. It would **not**
validate the protocol behind those numbers: `crosscheck/FINDINGS.md` F1.1–F1.6
documents five separate issues with how Table I is computed — most importantly
that its "Cross Validated" column is measured on data the model trained on, and
that naive persistence beats every model in that column.

Both facts can hold at once, and the report should state both.
