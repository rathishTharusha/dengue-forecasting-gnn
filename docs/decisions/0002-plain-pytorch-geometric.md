# 0002 — Plain PyTorch Geometric over `torch-geometric-temporal`

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Group 05

## Context

The reference implementation (Weng et al. / `disease_modeling_MLOS2`) builds its
spatio-temporal models — STGAT, A3TGCN, ASTGCN, DCRNN, AAGCN — on
`torch-geometric-temporal`. Reusing that library would be the fastest route to a baseline.

It is also fragile: it pins older torch/PyG versions, and Colab's preinstalled torch moves
underneath it, so installs break intermittently for reasons unrelated to our work. With six
people running notebooks on rotating Colab runtimes, that is a recurring tax.

More importantly, all three of our contributions attach *inside* the training loop and the
graph construction:

- physics-informed loss → the loss function,
- GAN augmentation → the batch sampler,
- adaptive adjacency → the graph itself, which must become a learned parameter.

A library that hides those behind opaque recurrent cells makes each contribution harder to
implement and harder to explain in the report.

## Decision

Build the Phase-1 baseline (`notebooks/baseline/`) on **plain PyTorch Geometric** — `GCNConv`
and `GATConv` with an explicit windowing, normalization, and training loop we own end to end.

Notebooks `00`–`02` are the exception: they reproduce Weng et al.'s published architectures
as published, so they keep the reference stack. Those notebooks exist to validate that we
reproduce the benchmark, not to be extended.

## Alternatives considered

| Option | Why not |
|---|---|
| `torch-geometric-temporal` throughout | Breaks often on Colab; hides exactly the seams our contributions attach to. |
| Raw PyTorch, no PyG | We would reimplement message passing for no benefit. `GCNConv`/`GATConv` are the uncontroversial parts. |
| DGL | No advantage here, and it would diverge from the reference implementation's conventions, complicating comparison. |

## Consequences

- The baseline is a plain, readable ~200-line pipeline: full control over windowing,
  normalization, loss, and adjacency.
- Data conventions still match the reference (target column index 5, `window = 3`,
  `horizon = 3`, adjacency + self-loops), so our numbers remain comparable to the published
  benchmark.
- We do not get the reference's temporal architectures for free. Phase-1 is GCN/GAT with a
  flattened 3-week window rather than a recurrent temporal encoder — a simpler model. Two
  notebook stacks now coexist, which is a small maintenance cost we accept.
- If a later phase needs a genuinely recurrent temporal encoder, implement it directly in
  `src/dengue_gnn/` rather than reintroducing the dependency.
