# Chapter 7 — The Five Architectures

*Prerequisites: Chapter 6.*

The benchmark paper evaluates five spatio-temporal GNNs. All five are reproduced
in this project, and all five are re-run under our own protocol in
`analysis/lib/reproduced.py`.

This chapter explains what each one computes, why it is built that way, and the
specific things that go wrong when you wire it up.

---

## 7.0 The common problem all five solve

Every architecture must handle **two** kinds of structure:

- **Spatial** — 25 districts, connected by a graph.
- **Temporal** — a 3-week window, ordered.

Chapter 6 handled space. For time there are three standard tools:

| tool | idea | cost |
|---|---|---|
| **RNN / LSTM / GRU** | carry a hidden state forward step by step | sequential, cannot parallelise over time |
| **1-D convolution** | slide a filter over the time axis | parallel, fixed receptive field |
| **Attention** | weight all timesteps by learned relevance | parallel, $O(T^2)$ |

The five architectures are essentially five different ways of combining a spatial
tool with a temporal one:

| architecture | spatial | temporal |
|---|---|---|
| STGAT | graph attention | LSTM |
| A3TGCN | GCN inside a GRU | GRU + attention over the window |
| ASTGCN | Chebyshev GCN + spatial attention | 1-D conv + temporal attention |
| DCRNN | diffusion convolution inside a GRU | GRU |
| AAGCN | adaptive GCN, two streams | 1-D conv |

### A note on LSTMs and GRUs

A plain RNN keeps a hidden state $h_t = \sigma(W[h_{t-1}, x_t])$. Over many steps
the gradient is multiplied by $W$ repeatedly, so it vanishes or explodes.

An **LSTM** adds a **cell state** with three gates: forget (what to discard),
input (what to add), output (what to expose). The cell state is updated
*additively*, so gradients flow back through many steps without repeated
multiplication. A **GRU** merges the gates into two, giving similar behaviour with
fewer parameters.

At $W = 3$, none of this really matters — three timesteps do not vanish
gradients. The recurrent machinery in these architectures is designed for long
sequences and is being applied to a very short one. Keep that in mind when
interpreting the results.

---

## 7.1 STGAT — graph attention plus LSTM

```python
self.gat   = GATConv(window, window, heads=8, dropout=dropout, concat=False)
self.lstm1 = nn.LSTM(n_nodes, hidden, num_layers=1)
self.lstm2 = nn.LSTM(hidden, hidden, num_layers=1)
self.head  = imp.make_head(hidden, horizon, inc, nodes=n_nodes)
```

**Flow:** GAT over the district graph → reshape → two stacked LSTMs → head.

```python
def forward(self, x, edge_index):
    batch, nodes, window = x.shape
    h = self.gat(x.reshape(batch * nodes, window),
                 self._batched_edges(edge_index, batch, nodes))
    h = torch.nn.functional.dropout(h, self.dropout, training=self.training)
    h = h.reshape(batch, nodes, window).movedim(2, 0)
    h, _ = self.lstm1(h)
    h, _ = self.lstm2(h)
    return self.head(h[-1])
```

**`concat=False`** averages the 8 attention heads instead of concatenating them,
so the output width stays `window` rather than becoming `8 × window`.

**`movedim(2, 0)`** puts the time axis first — `(batch, nodes, window)` becomes
`(window, batch, nodes)` — because PyTorch's LSTM expects `(seq, batch, features)`
by default. Getting this wrong does not error; it makes the LSTM treat districts
as timesteps.

**`h[-1]`** takes the final timestep's hidden state as the summary of the sequence.

### The unusual part

`nn.LSTM(n_nodes, hidden)` — the LSTM's **input width is the number of districts**.
Each timestep is a 25-vector of all districts at once. The node axis is
**collapsed into the feature axis**, so from this point the model has no per-node
representation; it has a single graph-level state.

That is why the head must emit all 25 nodes from one vector:

```python
# The LSTM has already collapsed the node axis, so the head emits every
# node from one graph-level vector -- one layer, as in the reference.
self.head = imp.make_head(hidden, horizon, inc, nodes=n_nodes)
```

This is a real architectural commitment: after the LSTM, STGAT cannot treat
districts differently except through what the head learns.

### Two deliberate departures from the reference

Both are documented in the docstring so a reader can tell reproduction from
repair:

**1. A dropout bug, fixed.**

```python
F.dropout(x, self.dropout)          # reference: `training` defaults to True
```

`torch.nn.functional.dropout` defaults `training=True`. The reference never
passes it, so **dropout stays active at inference even after `model.eval()`** —
test predictions are randomly corrupted, and running the same evaluation twice
gives different numbers.

Our version passes `training=self.training`. This is a repair, not a
reproduction, and the distinction is stated because it changes the numbers.

**2. Batched edge index.** See §7.6.

---

## 7.2 A3TGCN — attention temporal GCN

```python
self.core = A3TGCN(in_channels=1, out_channels=hidden, periods=window)
self.head = imp.make_head(hidden, horizon, inc)
```

A3TGCN = **A**ttention **T**emporal **G**raph **C**onvolutional **N**etwork.

Internally: a GCN is placed *inside* a GRU cell — the gate computations use graph
convolutions instead of plain linear layers, so the hidden state is updated using
neighbours' states as well as the node's own. On top, an attention layer weights
the `periods` timesteps.

The attention is the "A3T" part and it is what distinguishes this from plain
TGCN: rather than taking only the last hidden state, it learns a weighted
combination across the window.

### The shape problem

```python
# A3TGCN wants (nodes, in_channels, periods) and carries no batch
# dimension, so the batch is folded into the node dimension against a
# block-diagonal graph.
batch, nodes, window = x.shape
big_edges = _STGAT._batched_edges(edge_index, batch, nodes)
h = self.core(x.reshape(batch * nodes, 1, window), big_edges)
return self.head(torch.relu(h).reshape(batch, nodes, -1))
```

`torch_geometric_temporal`'s A3TGCN has **no batch dimension**. Its API assumes
one graph snapshot at a time.

The trick: fold the batch into the node axis. $B$ graphs of $N$ nodes become one
graph of $B \times N$ nodes, with a **block-diagonal** adjacency so the copies stay
disjoint. Since there are no edges between blocks, this is *mathematically
identical* to looping over the batch — and vastly faster.

It has to be *verified* identical, not assumed. §7.6.

---

## 7.3 ASTGCN — attention-based spatio-temporal GCN

```python
self.core = ASTGCN(nb_block=2, in_channels=1, K=3,
                   nb_chev_filter=64, nb_time_filter=64, time_strides=1,
                   num_for_predict=feat, len_input=window,
                   num_of_vertices=n_nodes)
```

The most elaborate of the five. Each block applies, in order:

1. **Spatial attention** — an $N \times N$ matrix computed *from the input*, so the
   effective graph changes week to week.
2. **Chebyshev graph convolution** — see below.
3. **Temporal attention** — over the window.
4. **1-D temporal convolution**.

### Chebyshev convolution and `K=3`

Rather than one hop per layer, approximate a spectral graph filter with a
polynomial in the graph Laplacian:

$$g_\theta \star x \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) x$$

where $T_k$ are Chebyshev polynomials, computed by the recurrence
$T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$.

The practical meaning: **`K=3` gives each layer a 3-hop receptive field** in one
operation, without materialising $\hat{A}^2$ or $\hat{A}^3$. The recurrence is
what makes it cheap.

(A GCN layer is the special case $K = 1$ with a particular normalisation. This is
the general form.)

### The head seam problem

```python
# The reference sets num_for_predict=horizon, making the final convolution
# the prediction itself and leaving nothing to attach a head to. Widening
# it to `feat` and projecting gives the same seam as the other four; the
# projection is present in every arm, `base` included.
```

The reference's final convolution **is** the output. There is no intermediate
representation, so there is nowhere to attach a per-horizon or probabilistic head.

The fix is to set `num_for_predict=16` and project down. This adds one projection
relative to the reference — and it is **present in every arm, including the
baseline**.

That last clause is the whole point. Comparisons *between arms* remain controlled,
because the extra layer is in all of them. Comparisons of this ASTGCN against
`reproduction/` are **not** valid, and the docstring says so explicitly.

This is what disciplined experiment design looks like: when you must deviate,
deviate uniformly, and record which comparisons the deviation invalidates.

---

## 7.4 DCRNN — diffusion convolution recurrent network

```python
self.core = DCRNN(window, hidden, K=32)
```

A GRU whose gate computations are **diffusion convolutions** (Chapter 6.4) rather
than linear layers. Bidirectional random walks — forward and reverse — so a
directed graph's asymmetry is used.

**`K=32` is very large.** It means 32 diffusion steps, a 32-hop receptive field on
a 25-node graph — every node reaches every other node many times over. The
adjacency is effectively saturated.

This is the reference's setting and is kept for fidelity, but it is worth naming:
at this graph size, `K=32` means DCRNN is barely using the graph *structure* at
all. It is closer to a fully-connected mixing operation.

Same batch-folding trick as A3TGCN, and here it mattered most:

```python
# ... this is equivalent to looping and ~400x faster. Looping made DCRNN the
# bottleneck of the whole sweep (6 s per batch of 32 against 0.01-0.37 s
# for the others).
```

---

## 7.5 AAGCN — adaptive adjacency GCN

```python
self.core = AAGCN(in_channels=1, out_channels=channels, edge_index=edge_index,
                  num_nodes=n_nodes, stride=1, residual=True,
                  adaptive=adaptive, attention=attention)
```

Originally from skeleton-based action recognition, where joints form a graph.
Two-stream: a fixed adjacency plus a learned one, with an attention module and a
residual connection.

**AAGCN builds its adjacency inside `__init__`**, so the graph must be passed at
construction rather than at forward time. That is why `build()` has a special case:

```python
if name == "AAGCN":
    if edge_index is None:
        raise ValueError("AAGCN needs edge_index at construction time")
```

An explicit error, rather than a confusing failure deeper in.

### A latent bug in the upstream library

```python
# ``adaptive`` is exposed here, unlike in their repo where it is hard-wired
# False. It cannot be enabled at their ``out_channels=1``: PGT computes
# ``inter_c = out_channels // 4``, which is 0, and the adaptive branch builds a
# zero-width convolution.
```

The reference hard-wires `adaptive=False`. Turning it on at their
`out_channels=1` computes `inter_c = 1 // 4 = 0` and constructs a convolution with
**zero output channels** — silently producing an empty tensor.

Raising `out_channels` to 8 and projecting back makes the switch actually
reachable. And again: **the projection is present in both arms**, so enabling
adaptivity is the only difference between them.

This is directly relevant to Contribution (c). AAGCN already *has* an adaptive
adjacency; the paper's use of it simply never switches it on. That is worth
knowing before claiming a learned graph as novel.

---

## 7.6 The batching bug — worth its own section

Three of the five architectures need this, and it is the most instructive bug in
the project.

```python
@staticmethod
def _batched_edges(edge_index, batch, nodes):
    if batch == 1:
        return edge_index
    offsets = torch.arange(batch, device=edge_index.device) * nodes
    return (edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)).permute(1, 0, 2).reshape(2, -1)
```

**What it does.** `edge_index` is `(2, E)`. To batch $B$ graphs, replicate the
edges $B$ times, offsetting copy $b$ by $b \times N$, so copy $b$'s edges connect
nodes $bN \ldots bN+N-1$. The result is `(2, B*E)` describing one block-diagonal
graph.

**What went wrong.** The original code passed the unmodified 25-node `edge_index`
alongside batched input. Nodes 0–24 got their edges. Every node index above 24 —
i.e. **every graph in the batch after the first** — appeared in no edge at all.

No error. No warning. PyG accepts an edge index that simply does not mention most
nodes; those nodes just receive no messages.

The model trained. The loss decreased. The metrics were plausible. It was
effectively running a graph model on 1/32 of the data and an isolated-node model
on the rest.

**How it was caught and fixed.** By asserting the property that must hold:

```python
# verified: batched output == per-window output, bit-identical
```

Run a batch through, run each window individually, compare element by element.
They must match exactly. This is now pinned with a test.

**The tell.** Fixing it made things *faster* — DCRNN 6.0 s → 0.375 s per fold,
A3TGCN 0.373 s → 0.020 s. A correctness fix that also speeds things up 16× is
usually a sign the slow path was doing something wrong.

**The general lesson:** graph libraries silently accept shapes that are wrong. Any
time you reshape around a library's batching assumptions, verify equivalence
against the obvious slow version, and keep that check as a test.

---

## 7.7 What is taken from the reference, and what is not

The module docstring draws the line explicitly:

> **Taken:** the architectures, from `torch_geometric_temporal` at the version
> their `requirements.txt` pins, wired exactly as `Models/gnn_models.py` wires
> them.
>
> **Not taken:** their evaluation. […] Their training loop also has no early
> stopping and no validation-based model selection: the final epoch's weights are
> scored.

And therefore:

> Numbers from this module are **not comparable to Table I**, and **are** comparable
> to every other row of this project's ablation table.

Stating which comparisons a piece of code supports — and which it does not — is
more valuable than the code itself. It is the difference between a baseline you
can build on and a number that will be misquoted six months later.

### The uniform head seam

```python
self.head = imp.make_head(hidden, horizon, inc)
```

Every architecture ends at a node-wise feature tensor and hands it to the same
head factory. The reason:

> That seam is what makes `per_horizon_heads` and `probabilistic` testable at
> all: without it they could be wired into only some of the five, and an ablation
> row that silently covers three architectures out of five is worse than no row.

An ablation row that means different things in different columns is worse than a
missing row, because a missing row is visible.

---

## 7.8 Reproduction results

All five reproduce. Deviation from published values, worst case 7.3%:

| Model | CV column (vs `results.txt`) | Full Dataset (vs Table I) |
|---|---|---|
| AAGCN | −0.00% / −0.00% | +4.3% / +2.1% |
| DCRNN | −0.35% / −0.40% | −2.8% / −1.5% |
| STGAT | +1.64% / +1.45% | +3.2% / −4.3% |
| ASTGCN | +1.92% / +2.30% | +7.3% / +2.5% |
| A3TGCN | +3.39% / +2.71% | +5.1% / +3.5% |

AAGCN matches to four decimal places.

**These are honest numbers.** The paper reports what its code produces. Chapter 5
established that the issue is what those numbers *measure*, not whether they are
real.

---

## 7.9 Which architecture is best?

Under our own protocol, on genuinely held-out data, pooled RMSE:

```
ours:fixed         43.36
ours:adaptive      43.81
persistence        44.80
theirs:adaptive    45.00
their DenseGCN     45.47
```

**The spread across all five architectures and both baselines is under 2 RMSE, and
persistence sits in the middle of it.**

The correct conclusion is not "architecture X wins". It is that **architecture
choice is not where the variance is** on this dataset. Chapter 3 explains why: the
signal is dominated by autocorrelation, the graph adds +0.07, the covariates are
corrupted, and one fold's error is largely one bad week.

Note also what this refutes. An earlier premise held that the repo's Phase-2
baseline was "very poor" (RMSE 64.67) and should be deleted. Measured
head-to-head on identical folds, it scores 45.47 — **at the floor, not far below
it**. The 64.67 figure came from a *different protocol* (8 origins rather than 3),
not a worse model.

That is Trap 5 from Chapter 5, encountered in our own records rather than
someone else's paper.

---

*Next: [Chapter 8 — Code Walkthrough](08_code_walkthrough.md)*
