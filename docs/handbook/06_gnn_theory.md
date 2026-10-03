# Chapter 6 — Graph Neural Networks from Scratch

*Prerequisites: Chapter 4. Matrix multiplication.*

We derive the graph convolution from a plain question — "how do I let districts
share information?" — and arrive at the standard GCN formula without ever quoting
it. Then we cover attention, diffusion, and the learned adjacency this project
tested.

---

## 6.1 Graphs

A graph is a set of **nodes** connected by **edges**.

Here: 25 nodes, one per district. An edge joins two districts that share a border.

```
Colombo ── Gampaha ── Kalutara
   │           │
Ratnapura ── Kegalle
```

### The adjacency matrix

Represent it as an $N \times N$ matrix $A$ where $A_{ij} = 1$ if there is an edge
from $i$ to $j$:

```python
a = np.eye(n)                                  # start with self-loops
for district, neighbours in adj.items():
    for neighbour in neighbours:
        a[index[district], index[neighbour]] = 1.0
a = np.maximum(a, a.T)                          # symmetrise
```

**Self-loops** (`np.eye`) mean every node is its own neighbour. Without them a
node's updated representation is built *only* from its neighbours, discarding its
own state — which would be a disaster when a node's own history is the strongest
predictor (Chapter 3).

**Symmetrisation** repairs the two one-way edges from Chapter 3 D4.

### Why `A @ H` is message passing

Let $H$ be $N \times F$: one row per node, $F$ features each. Then

$$(AH)_i = \sum_{j} A_{ij} H_j = \sum_{j \in \mathcal{N}(i)} H_j$$

**Row $i$ of $AH$ is the sum of the feature vectors of $i$'s neighbours** (plus its
own, via the self-loop).

That is the entire mechanism. One matrix multiplication propagates information one
hop. Everything called "message passing" is a variation on this.

```python
h = torch.relu(a @ self.gc1(h))
```

Read it right to left: transform each node's features (`gc1`), aggregate over
neighbours (`a @`), apply a non-linearity (`relu`).

### Two hops

Apply it twice and information travels two hops:

$$A(AH) = A^2H$$

$A^2_{ij}$ counts the paths of length 2 from $i$ to $j$. **A $k$-layer GNN gives
each node a receptive field of $k$ hops.**

This project uses **two layers**. With mean degree 4.72 and 25 nodes, two hops
already reaches most of the country, and Chapter 3 D5 showed the graph adds only
+0.07 of correlation — so more depth buys reach into a signal that is not there.

---

## 6.2 Normalisation — deriving the GCN

Raw $A$ has a fatal flaw. Colombo has 6 neighbours, Jaffna has 1. Summing
neighbours gives Colombo a value ~6× larger, purely from degree. Stack two layers
and it is 36×. Activations explode for high-degree nodes and vanish for low-degree
ones.

**Fix attempt 1 — average instead of sum.**

$$D^{-1}A, \qquad D_{ii} = \sum_j A_{ij}$$

Every node now receives the *mean* of its neighbours. Better, but asymmetric: a
message from Jaffna (degree 1) to Kilinochchi (degree 5) is scaled by 1/5, while
the reverse is scaled by 1/1. The same edge carries different weight in each
direction.

**Fix attempt 2 — split the normalisation between both ends.**

$$\hat{A} = D^{-1/2} A D^{-1/2}, \qquad \hat{A}_{ij} = \frac{A_{ij}}{\sqrt{d_i d_j}}$$

```python
deg = a.sum(1)
d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
return d_inv_sqrt @ a @ d_inv_sqrt
```

Now each edge is scaled by the geometric mean of its endpoints' degrees, and the
matrix stays symmetric. This is the **symmetric normalised adjacency**, and

$$H' = \sigma\!\left(\hat{A} H W\right)$$

is the **Graph Convolutional Network** layer (Kipf & Welling, 2017). We have just
derived it from the requirement that degree not distort magnitude.

`np.maximum(deg, 1e-12)` guards against division by zero for an isolated node.
With self-loops every degree is ≥ 1, but the guard costs nothing and the invariant
is not enforced elsewhere.

**Why "convolution"?** An image convolution aggregates a pixel's fixed grid
neighbourhood with learned weights. A graph convolution aggregates a node's
neighbourhood — the same idea where "neighbourhood" is defined by edges rather
than by position. The generalisation is why the name stuck.

---

## 6.3 The complete model

```python
def forward(self, x, fixed):
    """(batch, nodes, window) -> (batch, nodes, horizon)."""
    a = self.adjacency(fixed)
    h = torch.relu(self.lin_in(x))     # (B, N, W)      -> (B, N, hidden)
    h = torch.relu(a @ self.gc1(h))    # transform, aggregate, activate
    h = self.drop(h)
    h = torch.relu(a @ self.gc2(h))    # second hop
    return self.head(h)                # (B, N, hidden) -> (B, N, horizon)
```

Only nine lines, so it is worth being precise about the shapes.

`x` is `(batch, nodes, window)`. `nn.Linear` acts on the **last** axis, so
`lin_in` maps each node's 3-week window to a 64-dimensional vector independently —
no mixing across nodes yet.

`a @ h` with `a` of shape `(N, N)` and `h` of shape `(B, N, hidden)` broadcasts:
PyTorch treats `h` as a batch of `(N, hidden)` matrices and multiplies each. This
is what mixes information across nodes. It is also the one line where a
transposed or mis-ordered adjacency silently produces a wrong-but-plausible model.

**A useful mental division:**

| stage | mixes across | line |
|---|---|---|
| `lin_in` | time (the window) | `self.lin_in(x)` |
| `gc1`, `gc2` | features, then space | `a @ self.gcN(h)` |
| `head` | features → horizon | `self.head(h)` |

The temporal model here is deliberately minimal — a linear map over a 3-week
window. The five architectures in Chapter 7 differ mainly in replacing that with
something more expressive.

---

## 6.4 Beyond fixed adjacency

### Graph attention (GAT)

GCN weights every neighbour by degree alone. GAT **learns** how much each
neighbour matters:

$$e_{ij} = \text{LeakyReLU}\!\left(\vec{a}^\top [Wh_i \,\|\, Wh_j]\right), \qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)$$

Concatenate the two nodes' transformed features, score the pair, softmax over each
node's neighbourhood so weights sum to 1.

The attention weights are **input-dependent**: the same edge can carry different
weight for different weeks. That is the appeal — during an outbreak in Gampaha,
Colombo might attend to Gampaha strongly, and ignore it otherwise.

**Multi-head attention** runs $K$ independent attention mechanisms and
concatenates. One head may be unlucky in initialisation; averaging several is more
stable. This is the same construction as in Transformers.

**GAT still only attends over existing edges.** If the true coupling runs between
two non-adjacent districts, no amount of attention finds it.

### Diffusion convolution

Instead of one hop per layer, aggregate several hop-distances at once:

$$H' = \sum_{k=0}^{K} \hat{A}^{k} H W_k$$

Each power gets its own weight matrix, so the model can weight 1-hop, 2-hop and
3-hop neighbourhoods differently in a single layer.

The probabilistic reading is a random walk: $\hat{A}^k_{ij}$ is the probability of
reaching $j$ from $i$ in $k$ steps. **DCRNN** (Chapter 7) uses a bidirectional
version — forward and reverse walks — which matters on directed graphs.

### Adaptive adjacency — Contribution (c)

Why use a hand-built graph at all? Learn it.

$$\tilde{A} = \text{softmax}\!\left(\text{ReLU}(E_1 E_2^\top)\right)$$

with $E_1, E_2 \in \mathbb{R}^{N \times d}$ learned node embeddings.

```python
def adjacency(self, fixed):
    """Return the adjacency this arm uses, shape (n_nodes, n_nodes)."""
    if self.graph_mode == "none":
        return torch.eye(self.n_nodes, device=fixed.device)
    if self.graph_mode == "fixed":
        return fixed
    learned = torch.softmax(torch.relu(self.emb_src @ self.emb_dst.T), dim=-1)
    if self.graph_mode == "adaptive":
        return learned
    return self.alpha * fixed + (1.0 - self.alpha) * learned
```

Each piece has a job:

- **`emb_src @ emb_dst.T`** — an $N \times N$ score matrix. Two embeddings rather
  than one so the result can be **asymmetric**: influence need not be mutual.
- **`relu`** — clamps negative scores to zero, producing **sparsity**. Without it
  every pair is connected and the graph is dense noise.
- **`softmax(dim=-1)`** — each row sums to 1, so every node distributes a fixed
  budget of attention. This is the row-normalisation that keeps magnitudes stable.

$d = 16$ here, so the learned graph has $2 \times 25 \times 16 = 800$ parameters —
against $25^2 = 625$ for a free matrix. The factorisation is not saving
parameters; it is imposing **low-rank structure**, which is a regulariser: nodes
with similar embeddings behave similarly.

**Four modes, one encoder.** `none` (identity — 25 independent series, no message
passing), `fixed`, `adaptive`, and `hybrid` (`0.5 × fixed + 0.5 × adaptive`, which
is how Graph WaveNet actually uses it: the learned graph *augments* the prior
rather than replacing it).

The `none` arm is the control that matters. It answers "does the graph earn its
keep at all?" — a question the published work never asks.

---

## 6.5 The result

The adaptive graph was tested under the frozen protocol. Paired against `fixed`
(same folds, same seeds), in RMSE where **positive means worse**:

| arm | Δ RMSE vs fixed | p |
|---|---|---|
| `adaptive` | **+0.46** | 0.09 |
| `hybrid` | +0.19 | 0.37 |
| `none` | **−0.54** | 0.49 |

Read carefully:

- The learned graph is **not better**. If anything it is slightly worse.
- Removing the graph entirely (`none`) is **not worse** — the point estimate is
  actually a small improvement, at p = 0.49, i.e. indistinguishable from zero.

And the decisive evidence: the sign of the adaptive effect **flips between two
implementations** on identical folds and seeds — −0.47 in one, +0.45 in the other.

**An effect whose sign depends on implementation details is a null effect.**

This does not contradict Chapter 3 D5; it confirms it. Neighbours correlate at
0.62, non-neighbours at 0.55 — the graph carries +0.07 above a national common
mode, and a model with each district's own history already has most of that.

**The honest conclusion: on this dataset, at this resolution, the graph does very
little, and learning it does not help.** That is a finding, and it is stated as
one rather than buried.

---

## 6.6 Practical notes

### Batching for PyG models

Libraries like PyTorch Geometric represent graphs as an **edge index** — a
`(2, E)` tensor of source and destination indices — rather than a dense matrix.
Batching then requires care.

To batch $B$ graphs of $N$ nodes, treat them as one graph of $B \times N$ nodes
with a **block-diagonal** structure: offset graph $b$'s indices by $b \times N$.

**A bug found in this project:** a 25-node edge index was passed alongside batched
input. Nodes 0–24 got their edges; every node above 24 — i.e. every graph after
the first — got **none**. No error, no warning. The model trained, converged, and
produced plausible numbers, silently using only the first graph in each batch.

The fix and its verification:

```python
# verified: batched output == per-window output, bit-identical
```

Once corrected, the batched path was **also 16× faster** (DCRNN 6.0 s → 0.375 s
per fold; A3TGCN 0.373 s → 0.020 s), because it stopped falling back to
one-window-at-a-time processing.

**The general lesson:** when a graph library silently accepts a shape, verify that
the batched result equals the looped result *element by element*, and pin it with
a test. Speed improvements that arrive alongside a correctness fix are usually a
sign that the slow path was doing something wrong.

### Sparse vs dense

At $N = 25$, a dense $25 \times 25$ matrix multiply is trivially cheap and dense
`a @ h` is the clearest code. At $N = 10{,}000$ a dense matrix is 100 M entries
and you must use sparse operations. This project uses dense because it can, and
because clarity is worth more here than generality.

---

## 6.7 Summary

1. `A @ H` sums neighbour features. That is message passing.
2. $\hat{A} = D^{-1/2}AD^{-1/2}$ prevents degree from distorting magnitude — the
   GCN layer follows from that requirement.
3. Self-loops are essential when a node's own history is the strongest predictor.
4. $k$ layers = $k$ hops of receptive field.
5. GAT learns input-dependent edge weights, but only over existing edges.
6. Diffusion convolution aggregates multiple hop-distances at once.
7. An adaptive adjacency learns the graph — **and on this dataset it does not
   help.**
8. Always include a `none` arm. If the graph cannot beat no-graph, the graph is
   not the contribution.

---

*Next: [Chapter 7 — The Five Architectures](07_architectures.md)*
