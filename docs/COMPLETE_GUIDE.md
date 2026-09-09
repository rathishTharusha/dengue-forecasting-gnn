# The Complete Guide to This Project

**A from-scratch teaching document for dengue forecasting with graph neural networks.**

Audience: someone with basic coding knowledge (you can write a Python loop and a
function) and no assumed background in machine learning, graph theory, epidemiology,
or experimental statistics. By the end you should be able to rebuild this entire
project from an empty directory using only Python documentation.

This document explains *what* was built, *why* each decision was made, *how* each
piece works mathematically, and *what the code does* line by line. Where the project
got something wrong, that is documented too — the mistakes are the most instructive
part, and several of them are mistakes that also appear in published papers.

---

## How to read this

Eight parts. They build on each other; read in order the first time.

| Part | What it covers | Read this if you want to know… |
|---|---|---|
| I | The problem and the data | What we predict, and from what |
| II | Machine learning from first principles | How neural networks actually work |
| III | Graphs and graph neural networks | Why a *graph* network, and what that means |
| IV | The evaluation protocol | Why most of the engineering effort went here |
| V | The target parameterisation | The single most important modelling decision |
| VI | The code, module by module, function by function | How to write it yourself |
| VII | Every experiment and what it found | What is actually true about this problem |
| VIII | Rebuilding from scratch, pitfalls, glossary | Your implementation checklist |

Boxes marked **⚠ Trap** are mistakes this project actually made and had to fix. They
are the parts most worth internalising.

---

# Part I — The Problem and the Data

## 1.1 What we are trying to do

Dengue is a mosquito-borne viral disease. In Sri Lanka it is endemic: cases occur
every year, rising and falling with the monsoons, and occasionally exploding into
large outbreaks. Health authorities have finite resources — mosquito-control teams,
hospital beds, blood products. If you know three weeks in advance that Colombo is
heading into an outbreak, you can move resources there before it happens.

So the task is:

> Given everything observed up to week *t*, predict dengue cases in each of Sri
> Lanka's 25 districts for weeks *t+1*, *t+2* and *t+3*.

That is **multi-horizon, multi-site time-series forecasting**.

- **Multi-horizon** — we predict 3 steps ahead, not 1. The horizon is *H = 3*.
- **Multi-site** — 25 districts at once, not one.
- **Time-series** — the data has a time order, and that order is sacred. You may
  never use the future to predict the past.

Why a *graph* neural network is a candidate: districts are not independent. People
move between them carrying the virus; neighbouring districts share climate; an
outbreak in Gampaha is informative about Colombo next door. A model treating the 25
districts as 25 unrelated problems throws that information away.

## 1.2 The data

One file: `notebooks/baseline/sri_lanka_2013-2022_shifted.npy`, a NumPy array of
shape **(459, 25, 11)**.

```
axis 0  →  459 weeks       (time),   2013 through 2022
axis 1  →   25 districts   (space),  alphabetical order
axis 2  →   11 features    (channels)
```

So `raw[t, n, f]` is feature *f* in district *n* during week *t*.

**Feature index 5 is the weekly dengue case count.** That is what we predict. The
other ten channels are covariates — meteorological and geospatial measurements that
plausibly drive mosquito populations:

- precipitation
- air temperature
- land-surface temperature
- humidity
- NDVI (Normalised Difference Vegetation Index — a satellite greenness measure, a
  proxy for vegetation and standing water)
- soil moisture
- canopy cover

These come from NASA EarthData satellite products aggregated to weekly resolution per
district. Case counts come from Sri Lanka's Ministry of Health / Epidemiology Unit
weekly reports.

### 1.2.1 The covariates are already lag-shifted

Critical, and easy to get wrong. The filename ends in `_shifted` because each
covariate has already been moved forward in time by its empirically optimal delay.

Why: rain does not cause dengue cases this week. Rain creates standing water; water
breeds mosquitoes; mosquitoes mature; an infected mosquito bites someone; the person
incubates the virus; symptoms appear; they visit a clinic; the clinic reports it.
That chain takes roughly **12 weeks** for precipitation and about **17 weeks** for
minimum NDVI.

So the array already contains, at week *t*, the precipitation from week *t − 12*.

> **⚠ Trap.** Do not apply a second lag. Shift again and you are looking at rainfall
> from 24 weeks ago — nearly two dengue seasons back, carrying much less signal. This
> is project rule **D9**.

### 1.2.2 The graph

`notebooks/baseline/sri_lanka_adj_list.json` maps each district to its geographic
neighbours:

```json
{"Colombo": ["Gampaha", "Kalutara", "Ratnapura"], "...": ["..."]}
```

Verified facts (recomputed from the file, not quoted):

- 25 nodes (districts)
- **116 directed edges** between distinct districts
- **141 directed edges** once self-loops are added (116 + 25)
- Each district has between **1 and 9** neighbours

Edges encode *geographic adjacency*, built from district-to-district distances.
Ideally you would use human mobility data — actual counts of people travelling
between districts — because that is what moves the virus. That data is unavailable
for Sri Lanka at this resolution, so physical adjacency is the proxy. Replacing this
hand-built graph with a *learned* one is one of the project's contributions.

### 1.2.3 The target distribution — the facts that drive every design decision

All recomputed directly from the array:

| Statistic | Value | Why it matters |
|---|---|---|
| Median weekly cases per district | **13** | A typical week is small |
| Mean weekly cases per district | **43.98** | Mean ≫ median ⇒ heavily right-skewed |
| Maximum weekly cases | **2,631** | The tail is 200× the median |
| Zero-case district-weeks | **9.74 %** | One week in ten has *no* cases |
| Lag-1 autocorrelation, mean per district | **0.679** | This week strongly predicts next week |
| Lag-1 autocorrelation, all district-weeks pooled | **0.921** | Inflated — see below |

> **A precision point worth absorbing.** The pooled figure (0.921) and the
> per-district figure (0.679) measure different things. Pooling all 25 districts into
> one scatter plot puts Colombo's consistently-high values in the top-right and a
> small district's consistently-low values in the bottom-left, so the correlation
> partly measures *"big districts are big"* rather than *"this week predicts next
> week."* The per-district mean removes that. Whenever you see an autocorrelation
> quoted, ask which one it is. `docs/DATA.md` quotes 0.68 — the per-district figure,
> and the honest one.

Three consequences follow, and they determine nearly everything downstream:

**(a) Because 9.74% of observations are zero, you cannot use MAPE.** Mean Absolute
Percentage Error divides by the true value; divide by zero and you get infinity. We
report SMAPE (symmetric, bounded) plus a zero-masked MAPE. See §4.4.

**(b) Because the tail is 200× the median, you cannot train on raw counts.** Squared
error on raw counts is dominated by a handful of outbreak weeks; the model optimises
for those and ignores everything else. We train in `log1p` space. See §5.1.

**(c) Because per-district lag-1 autocorrelation is 0.68, "next week looks like this
week" is a strong forecast.** This is the single most important fact in the project.
It means the naive baseline is *hard to beat*, and it means the model should predict
the *change* from this week rather than the absolute level. See §5.2.

---

# Part II — Machine Learning From First Principles

If you already know backpropagation, skim to Part III. Otherwise this part gives you
exactly the machinery this project uses and nothing more.

## 2.1 Supervised learning in one paragraph

You have inputs *x* and correct answers *y*. You want a function *f* mapping *x* to
*y*. You pick a family of functions with adjustable knobs (**parameters**, *θ*),
define a number saying how wrong the current setting is (the **loss**), then adjust
the knobs to make that number smaller. That is all machine learning is. The
interesting questions are: which family, which loss, and how you adjust.

## 2.2 Tensors

A **tensor** is an n-dimensional array.

- 0-D: a single number — a *scalar*
- 1-D: a list of numbers — a *vector*
- 2-D: a table — a *matrix*
- 3-D: a stack of tables, and so on

PyTorch tensors are NumPy arrays with two extra abilities: they can live on a GPU,
and they **remember how they were computed** so gradients can be traced backward.

The `.shape` is the length of each axis. Nearly every bug in this kind of code is a
shape bug, so the convention throughout this codebase is to write the expected shape
in the docstring of every tensor-taking function:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Args: ``(batch, n_feat, window)``. Returns ``(batch, hidden)``."""
```

Copy that habit. It costs one line and saves hours.

## 2.3 The linear layer

The fundamental building block:

```
y = x W + b
```

`x` has width *d_in*, `W` is a *d_in × d_out* parameter matrix, `b` is a *d_out*
parameter vector, `y` has width *d_out*. In PyTorch:

```python
layer = nn.Linear(d_in, d_out)   # creates W and b, randomly initialised
y = layer(x)
```

This is a **learned linear map**: it can rotate, scale, project, and mix features in
any linear combination. What it cannot do is anything non-linear — and stacking two
linear layers gives another linear layer (*x W₁ W₂* is just *x W₃*), so stacking
alone buys nothing.

## 2.4 Activation functions

To get non-linearity you insert a function that bends the output. This project uses
four:

**ReLU** — `relu(x) = max(0, x)`. Zeroes negatives, passes positives unchanged. Cheap,
and its gradient is 0 or 1, avoiding the vanishing-gradient problem of older
activations.

**LeakyReLU** — `max(αx, x)` with small α (0.2 here). Like ReLU, but negatives leak
through scaled down instead of dying. Used in the attention and GAN code, where a
hard zero would kill gradients.

**Sigmoid** — `σ(x) = 1 / (1 + e^{−x})`. Squashes any real number into (0, 1). Used
whenever you want a **gate** — a number saying "how much of this to let through."

**tanh** — squashes into (−1, 1). Used where you want a *signed* magnitude.

The sigmoid-and-tanh pairing appears twice in this project (gated convolutions, LSTM
cells) and always means the same thing: *tanh proposes an update, sigmoid decides how
much of it passes.*

## 2.5 Dropout

During training, randomly zero a fraction *p* of activations. During evaluation, do
nothing.

Why it helps: the network cannot rely on any single feature, because that feature
might vanish. It is a **regulariser** — it fits training data slightly worse in
exchange for generalising better. This project uses *p = 0.1*.

> **⚠ Trap.** Dropout must be *off* at evaluation. In PyTorch, call `model.eval()`
> before evaluating and `model.train()` before training. Forget it and your validation
> numbers are noise. `_evaluate` in `experiment.py` calls `model.eval()` as its first
> statement and `train_fold` calls `model.train()` at the top of every epoch. That is
> not decoration.

## 2.6 Loss functions

The loss is one number measuring wrongness. The data loss here is **mean squared
error**:

```
MSE = mean( (prediction − truth)² )
```

Squaring makes errors positive and penalises large errors disproportionately (error
10 costs 100; error 1 costs 1). That property is exactly why the *space* you compute
it in matters so much — see §5.1.

## 2.7 Gradient descent and backpropagation

The loss is a function of the parameters. Calculus gives the **gradient** — the vector
of partial derivatives ∂loss/∂θ — pointing in the direction of steepest *increase*.
To reduce the loss, step the opposite way:

```
θ ← θ − η · ∂loss/∂θ
```

*η* (eta) is the **learning rate**, the step size. This project uses *η = 10⁻³*.

**Backpropagation** computes that gradient efficiently, using the chain rule to push
derivatives backward layer by layer. PyTorch does it automatically: calling
`loss.backward()` fills every parameter's `.grad` field.

The training step is always these lines:

```python
opt.zero_grad()      # clear gradients from the previous step (they accumulate!)
loss.backward()      # compute new gradients
# (optionally clip them here)
opt.step()           # apply the update
```

> **⚠ Trap.** Omitting `zero_grad()` makes gradients accumulate across steps, silently
> training a different and wrong model. It does not crash.

## 2.8 Adam and weight decay

Plain gradient descent uses one learning rate for every parameter. **Adam** adapts a
per-parameter rate from running averages of the gradient and its square. It converges
faster and needs less tuning. This project uses `lr=1e-3, weight_decay=5e-4`.

**Weight decay** adds a penalty proportional to weight size, pushing weights toward
zero — another regulariser, preferring simpler models.

## 2.9 Gradient clipping

If a gradient is enormous, one step can throw the parameters somewhere terrible.
**Clipping** rescales the gradient vector so its norm is at most a threshold:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```

This project clips at 5.0, added specifically because the model occasionally produced
hallucinated case spikes.

## 2.10 Train / validation / test, and early stopping

Split three ways:

- **Train** — the model fits these.
- **Validation** — the model never fits these, but *you* use them to decide when to
  stop and which hyper-parameters to pick.
- **Test** — touched once, at the very end, to report a number.

**Early stopping**: after each epoch, score on validation. If improved, save a copy of
the weights. If not improved for `patience` epochs, stop and restore the best copy.
This project uses `patience=25`, `epochs=120`.

From `train_fold`:

```python
if val_rmse < best - 1e-4:
    best = val_rmse
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
else:
    wait += 1
    if wait >= cfg.patience:
        break
```

Two details worth copying. The `- 1e-4` means a microscopic improvement does not count,
so the patience counter is not reset by noise. And `.detach().cpu().clone()` makes a
genuine independent copy — without `.clone()` you store a *reference* that keeps
changing as training continues, so "restoring the best weights" restores the final
weights instead. Silent, and catastrophic.

> **⚠ Trap.** The test set is not a second validation set. If you look at the test
> score, change something, and look again, you have leaked test information into your
> decisions and your reported number is optimistically biased. This is the most common
> way honest people produce dishonest results.

---

# Part III — Graphs and Graph Neural Networks

## 3.1 What a graph is

A **graph** is a set of **nodes** connected by **edges**. Here: 25 districts (nodes)
joined when they share a border (edges).

The standard representation is an **adjacency matrix** *A*, an *N × N* table where
`A[i][j]` is non-zero if node *i* is connected to node *j*. For 25 districts that is a
25 × 25 matrix, mostly zeros (only 116 of 600 possible off-diagonal entries are set).

**Self-loops.** We add the identity matrix, setting `A[i][i] = 1`. Without this, a
district's own history would be excluded when computing its own next value — clearly
wrong, since the strongest predictor of Colombo next week is Colombo this week.

## 3.2 Normalisation, and why row-normalisation here

Raw adjacency has a problem: a district with 9 neighbours sums 9 contributions while a
district with 1 neighbour sums 1. Activations then scale with node degree, which is an
artefact of geography, not signal.

**Row normalisation** divides each row by its sum, so each row sums to 1:

```python
def row_normalize(adj, eps=1e-8):
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(eps)
```

Now each district's incoming influence is a **weighted average** of its neighbours,
comparable regardless of how many it has. (`clamp_min(eps)` guards against dividing by
zero for an isolated node.)

There is another common choice, **symmetric normalisation**, *D^(−1/2) (A+I) D^(−1/2)*,
used by the classic Kipf & Welling GCN and by PyTorch Geometric's `GCNConv`. It is not
wrong — it is just different, and it matters enormously that you know which you used:

> **⚠ Trap — and this is finding F6, a real one in this project.** The Phase-2 paper
> compared a dense adaptive model against a PyTorch Geometric `GCNConv` baseline and
> attributed the whole RMSE difference to the learned adjacency. But `GCNConv` uses
> symmetric normalisation while the dense path used row normalisation. **Two things
> changed at once**, so the comparison attributed nothing. The fix was to build a
> control that shares the propagation path and differs *only* in the adjacency —
> which is what `AdaptiveGCN(use_adaptive=False)` is for. This is now project rule
> **D7: one variable at a time.**

## 3.3 Graph convolution

The core operation. Each node's new representation is a weighted average of its
neighbours' representations, then a learned linear map and a non-linearity:

```
H = ReLU( A · X · W )
```

- `X` is *(N, F)* — each node's features
- `W` is *(F, F')* — learned, shared by every node
- `A` is *(N, N)* — the normalised adjacency
- `A · X` mixes across **nodes** (spatial)
- `· W` mixes across **features** (channels)

Stacking two of these means information travels two hops: after layer 1, each node
knows about its neighbours; after layer 2, about its neighbours' neighbours. That
distance is the **receptive field**.

This project uses exactly two layers:

```
H1 = ReLU(A @ X  @ W1)
H2 = ReLU(A @ H1 @ W2)
Y  = Linear(Dropout(H2))
```

Why only two: with 25 nodes and a graph of diameter around 6, three or more layers
start averaging most of the country into every node — a failure mode called
**over-smoothing**, where all node representations converge to the same vector and the
model loses the ability to distinguish districts at all.

### 3.3.1 The batch dimension, and `einsum`

In code you process a batch of windows at once, so `X` is *(B, N, F)* and you need to
multiply an *(N, N)* matrix against it without touching the batch axis. That is what
`einsum` expresses:

```python
h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w1(x)))
```

Read the subscript string as: *"take `adj` indexed by (i,j) and `w1(x)` indexed by
(b,j,f); sum over the repeated index `j`; produce output indexed by (b,i,f)."* The
summed index `j` is the neighbour being aggregated. `b` and `f` pass through
untouched.

`einsum` is worth learning properly. It replaces a stack of `reshape`/`transpose`/
`bmm` calls with one readable line, and — because you name every axis — it is far
harder to get silently wrong.

## 3.4 The adaptive adjacency — Contribution (c)

Geographic adjacency is a guess. Two districts may be strongly coupled by a highway
without sharing a border; two bordering districts may be separated by mountains. So:
**learn the graph instead**.

Following Graph WaveNet (Wu et al., 2019), create two free embedding matrices *E₁* and
*E₂*, each *(N, d)*, and define:

```
A_adp = softmax( ReLU( E₁ E₂ᵀ ) )
```

Step by step:

1. `E₁ E₂ᵀ` — *(N,d) × (d,N) = (N,N)*. Every ordered district pair gets a score, the
   dot product of one district's "source" embedding with another's "target" embedding.
   Because it is **asymmetric** (E₁ ≠ E₂), the model can learn that *i* influences *j*
   without *j* influencing *i* — appropriate for directional flows of people.
2. `ReLU` — zeroes negative scores, making the learned graph **sparse** rather than
   fully connected. Without it every pair would have some weight.
3. `softmax(dim=1)` — makes each row sum to 1, matching `row_normalize`, so the learned
   and fixed graphs are on the same scale and can be blended.

In code, remarkably small:

```python
class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes, emb_dim=10, init_scale=0.1):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * init_scale)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * init_scale)

    def forward(self):
        return F.softmax(F.relu(self.e1 @ self.e2.t()), dim=1)
```

`nn.Parameter` is the important part: it tells PyTorch these tensors are trainable, so
they appear in `model.parameters()` and receive gradients. The whole learned graph
costs *2 · N · d* = 2 · 25 · 10 = **500 parameters**.

### 3.4.1 The gate

Rather than replacing geography, **blend** it with the learned graph using a single
learned scalar *g*:

```
A_blend = σ(g) · A_fixed + (1 − σ(g)) · A_adp
```

`σ(g)` lies in (0,1), so this is a convex combination — a dial between pure geography
(σ=1) and the pure learned graph (σ=0). One parameter.

`gate_init = 1.5` gives σ(1.5) ≈ **0.82**, so training *starts* leaning on geography
and shifts toward the learned graph only if the data supports it. That is a deliberate
prior: geography is a decent guess, so make the model earn its departure from it.

The gate is also a **diagnostic**. `gate_value()` is logged in every result row, so you
can answer "did the model actually use the learned graph?" from the CSV rather than by
guessing.

```python
def blended_adjacency(self):
    if not self.use_adaptive:
        return self.adj_fixed
    gate = torch.sigmoid(self.gate)
    return gate * self.adj_fixed + (1.0 - gate) * self.adaptive()
```

Note `use_adaptive=False` returns `A_fixed` and creates **no** embedding or gate
parameters at all. That is the control arm — genuinely identical except for the one
variable.

### 3.4.2 Buffers versus parameters

```python
self.register_buffer("adj_fixed", adj_fixed.clone().float())
```

A **buffer** is a tensor that is part of the model's state (it moves with
`.to(device)`, it is saved in `state_dict`) but is **not trained**. Geography is data,
not something to learn. Using `nn.Parameter` here would let gradient descent quietly
rewrite the map of Sri Lanka.

## 3.5 Graph attention (GAT)

An alternative to fixed weights: let the model compute edge weights **from the node
features, freshly for every input**. Following Veličković et al. (2018):

For each pair (*i*, *j*):

1. Project both nodes: `h = x W`
2. Score the pair: `e_ij = LeakyReLU(a_src · h_i + a_dst · h_j)`
3. Normalise over *i*'s neighbours: `α_ij = softmax_j(e_ij)`
4. Aggregate: `h'_i = Σ_j α_ij h_j`

**Masking is essential.** Before the softmax, set the score to −∞ for any pair that is
not an edge. `exp(−∞) = 0`, so non-neighbours get exactly zero weight:

```python
mask = (adj_mask > 0).unsqueeze(0).unsqueeze(-1)
logits = logits.masked_fill(~mask, float("-inf"))
alpha = F.softmax(logits, dim=2)
alpha = torch.nan_to_num(alpha)   # an isolated node has an all -inf row
```

That `nan_to_num` is a real edge case: if a node had *no* neighbours at all, every
logit in its row is −∞, softmax gives 0/0 = NaN, and a single NaN poisons the entire
backward pass. Since we add self-loops it cannot happen here — but the guard costs
nothing and the failure it prevents is a whole training run silently producing NaN.

**Multi-head attention** runs *K* independent attention mechanisms and combines them,
so different heads can specialise. This implementation **averages** heads:

```python
out = torch.einsum("bijk,bjkf->bikf", alpha, h)
return out.mean(dim=2)  # average heads; concatenating would change width
```

The comment matters. The original paper *concatenates* heads in intermediate layers,
which multiplies the output width by *K*. Averaging keeps width fixed at `out_dim`, so
the GAT arm is a drop-in replacement for the GCN arm and the comparison stays
attributable (D7 again).

### 3.5.1 GAT versus adaptive adjacency — a genuinely different hypothesis

Both learn edge weights. The difference is what they are a function of:

| | Adaptive adjacency | Graph attention |
|---|---|---|
| Weights depend on | Nothing — free parameters | The current input features |
| After training | One fixed graph | A different graph every forward pass |
| Can add edges? | **Yes** — any pair can get weight | **No** — masked to existing edges |
| Parameters | 2·N·d = 500 | Scales with feature width |

They encode different beliefs about what geography gets wrong. The adaptive graph says
*"the map is missing edges"*. Attention says *"the map is right, but the strength of
each link changes with conditions."* Running both is how you find out which.

## 3.6 Temporal operators

A window of 3 weeks is a short sequence, but it *is* a sequence, and the order matters.
The project offers four ways to handle it.

### 3.6.1 The original approach — and its flaw

The first version flattened `(window, features)` into one vector of length 33 and
applied a linear layer.

> **⚠ Trap.** That treats week *t−3* and week *t−1* as unrelated columns. Nothing in
> the architecture knows they are ordered. The "spatio-temporal" model **had no
> temporal component at all** — which is the most likely reason a plain LSTM
> outperformed it. If you call a model spatio-temporal, be able to point at the part
> that is temporal.

### 3.6.2 Gated dilated causal convolution (`GatedTCN`)

Graph WaveNet's temporal block. Two parallel 1-D convolutions form a gate:

```
h = tanh(W_f * x) ⊙ sigmoid(W_g * x)
```

The filter branch proposes an update; the gate branch decides how much passes. `⊙` is
elementwise multiplication.

**Causality** is enforced by left-padding:

```python
self.pad = (kernel - 1) * dilation
x = F.pad(x, (self.pad, 0))     # pad on the LEFT only
```

Padding only on the left means the output at week *t* never sees week *t+1*. Padding
symmetrically — the default in most convolution code — would let the model peek at the
future. **That is data leakage inside the architecture**, and it is invisible in
training curves; it just makes results look wonderful and generalise terribly.

Only the final timestep is returned:

```python
return out[..., -1]
```

That is the position whose receptive field covers the whole window, and the prediction
head wants one vector per node.

**Dilation** spaces the convolution taps apart (dilation 2 looks at *t*, *t−2*, *t−4*)
to grow the receptive field without more parameters. With *W = 3* there is no room, so
the default is 1 — the parameter exists so the window can be widened later.

### 3.6.3 Recurrent encoders (`RecurrentTemporal`)

A **recurrent neural network** processes a sequence one step at a time, carrying a
hidden state:

```
h_t = f(x_t, h_{t−1})
```

**LSTM** and **GRU** are RNN variants with gates that control what to keep and what to
forget, which lets them learn longer dependencies than a plain RNN. You do not need to
implement them — `nn.GRU` and `nn.LSTM` are built in.

```python
out, _ = self.rnn(x.transpose(1, 2))  # (batch, window, hidden)
if not self.attention:
    return out[:, -1]
```

The `transpose(1, 2)` is needed because PyTorch RNNs with `batch_first=True` expect
*(batch, sequence, features)*, whereas convolutions expect *(batch, features,
sequence)*. Same data, different axis order. Getting this wrong does not crash — it
silently treats features as timesteps.

**Temporal attention** (A3TGCN's idea): instead of taking the last hidden state,
learn a score for each timestep and take a weighted sum:

```python
w = F.softmax(self.score(out), dim=1)  # (batch, window, 1)
return (w * out).sum(dim=1)
```

Now the model can emphasise whichever weeks matter for the horizon rather than being
forced to squeeze everything into the final state.

### 3.6.4 Composing architectures

The point of having a `spatial` axis and a `temporal` axis is that published
architectures become **compositions** rather than separate codebases:

| Name | `spatial` | `temporal` |
|---|---|---|
| GCN (control) | `gcn` | `none` |
| Graph WaveNet-style | `gcn` | `gtcn` |
| GAT | `gat` | `none` |
| STGAT | `gat` | `lstm` |
| A3TGCN | `gcn` | `gru_attn` |

Every arm then shares one propagation path, one target transform, one optimiser, one
evaluation. A difference between rows is attributable to the architecture alone.

> **A deliberate deviation, with a cost.** The library `torch_geometric_temporal`
> provides STGAT and A3TGCN off the shelf, and this project does **not** use it
> (decision ADR-0002). The cost is real: these are *our* renderings, so a gap could be
> an implementation difference rather than a genuine one. The benefit is that
> importing five upstream implementations would reintroduce exactly the confound
> finding F6 was about — five different normalisations, five different optimisers,
> five different target transforms. Both readings should be stated in any write-up.

---

# Part IV — The Evaluation Protocol

This is where most of the project's engineering effort went, and it is the part most
likely to be done badly elsewhere. **A model is only as trustworthy as the protocol
that measured it.** You can have a perfect implementation of a brilliant architecture
and produce a completely meaningless number.

## 4.1 Why you cannot use a random split

The standard machine-learning move is: shuffle your data, take 80% to train and 20% to
test. **On time series this is fatal.**

If you shuffle, week 300 can land in training while week 299 and week 301 land in test.
The model has effectively seen the answer. Because consecutive weeks correlate at 0.68,
your model looks superb and will fail completely in deployment, where the future is
genuinely unavailable.

This is **temporal leakage**, and it is non-negotiable. Never shuffle a forecasting
dataset.

## 4.2 Why a single chronological split is not enough either

The obvious fix is a single chronological split: train on the first 70%, test on the
last 20%. Phase 1 did exactly that, and it failed in an instructive way.

Across the hyper-parameter tuning grid, **validation and test RMSE were
anti-correlated: r = −0.74.** The configuration chosen on validation (best validation
RMSE 90.9) scored 67.2 on test — one of the worst in the grid — while three
configurations that would have *beaten* persistence on test (RMSE 53–56) were
discarded for poor validation scores.

Read that again, because it is the most important negative result in the project:
**picking the model that did best on validation was worse than picking at random.**

The reason is that the 45-week validation slice and the test slice fell in different
epidemic regimes — one quiet, one outbreak. "Best on a quiet period" carries no
information about "best on an outbreak." A single split gives you one sample of one
regime, and no way to know which you got.

## 4.3 Rolling-origin cross-validation

The fix is to evaluate at **many different points in time** and look at the
distribution, not a single number.

**Rolling-origin** (also called expanding-window) cross-validation:

1. Pick a point in the series — the **origin**.
2. Train on everything before it.
3. Test on a window just after it.
4. Move the origin forward and repeat.

Every fold respects time order. You get one score per origin, so you can compute a
mean, a spread, and a significance test.

### 4.3.1 The exact arithmetic

```python
def fold_split(raw, cfg, fold_idx):
    ids = list(range(cfg.window, raw.shape[0] - cfg.horizon))
    n = len(ids)
    origin = cfg.origins[fold_idx]
    cut  = int(origin * n)
    tend = int(min(origin + cfg.test_frac, 1.0) * n)
    train_pool, test_ids = ids[:cut], ids[cut:tend]
    return train_pool[: -cfg.val_weeks], train_pool[-cfg.val_weeks :], test_ids, origin
```

Line by line:

- `ids` are the valid **forecast origins** — week indices at which a window can be
  formed. It starts at `cfg.window` (you need 3 prior weeks of input) and stops
  `cfg.horizon` short of the end (you need 3 future weeks of ground truth). With
  T=459, W=3, H=3 that gives 453 usable indices.
- `origin` is a **fraction**, e.g. 0.55 means "55% of the way through".
- `cut` converts it to an index; `tend` is the end of the test window.
- Training is everything before `cut`. **The last `val_weeks` (30) of the training
  pool become validation** — chronologically the most recent training data, and
  therefore the closest thing to the test period.
- Test is `ids[cut:tend]`.

Running it on the real data produces exactly this:

| Fold | Origin | Train | Val | Test | Test week range |
|---|---|---|---|---|---|
| 1 | 0.400 | 151 | 30 | 34 | 184–217 |
| 2 | 0.475 | 185 | 30 | 34 | 218–251 |
| 3 | 0.550 | 219 | 30 | 34 | 252–285 |
| 4 | 0.625 | 253 | 30 | 34 | 286–319 |
| 5 | 0.700 | 287 | 30 | 34 | 320–353 |
| 6 | 0.775 | 321 | 30 | 34 | 354–387 |
| 7 | 0.850 | 355 | 30 | 34 | 388–421 |
| 8 | 0.925 | 389 | 30 | 34 | 422–455 |

Notice: the training set **grows** each fold (151 → 389) while validation and test stay
fixed in size. That is the "expanding window" part, and it mirrors reality — as time
passes you accumulate more history.

### 4.3.2 Why the test windows are exactly adjacent

Look at the test ranges: 184–217, then 218–251. They touch but never overlap. That is
deliberate, and it comes from an arithmetic constraint:

```
origins spaced by 0.075, and test_frac = 0.075   →   adjacent, disjoint windows
```

Phase 2 used 3 origins (0.55, 0.70, 0.85) with `test_frac=0.15`. Spacing 0.075 with
width 0.15 means consecutive test windows **overlap by 50%**.

Why that is a problem: a paired statistical test assumes the paired observations are
independent. Two folds sharing 57% of their test data are not two independent
observations — they are closer to one and a bit. Treating them as independent makes
your test **anti-conservative**: it reports significance you have not earned.

The cost of fixing this is a shorter test window per fold (34 weeks instead of 68).
That is a real cost, accepted deliberately, in exchange for folds a paired test can
legitimately treat as independent. The Phase-2 origins are a subset of the new grid, so
old folds remain identifiable in new results.

> This is project rule **D1: the protocol is frozen.** Once you change origins,
> horizon, or normalisation, no row of your results table can be compared to any other.
> Every ablation would need re-running. Freeze it early and write it down.

### 4.3.3 Seeds

Neural network training is stochastic: random initialisation, random dropout, random
data order. Train the same configuration twice and you get different numbers.

So each fold is run with **3 random seeds** (0, 1, 2). With 8 folds that is **24
observations** per configuration. Every one of those is written to disk separately.

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

Deterministic baselines (`seasonal_naive`, `ridge`) run **one** seed, not three,
because three identical fits would be duplicates that inflate the apparent sample size:

```python
DETERMINISTIC = {"seasonal_naive", "ridge"}
```

That is why the results table shows *n = 8* for those rows and *n = 24* for the rest.

## 4.4 Normalisation — from training statistics only

Neural networks train badly when features have wildly different scales. The fix is
**z-normalisation**: subtract the mean, divide by the standard deviation, so each
feature has mean 0 and SD 1.

The subtlety is **which** mean and standard deviation:

```python
train_weeks = raw[: train_ids[-1]]
fmean = train_weeks.reshape(-1, n_feat).mean(0)
fstd  = train_weeks.reshape(-1, n_feat).std(0) + 1e-6
feat  = (raw - fmean) / fstd
```

`train_weeks` is **only the training portion**. If you computed the mean over the whole
array, the test period's statistics would leak into training — a subtle but real form
of leakage, because knowing the future mean tells you something about the future.

The `+ 1e-6` prevents division by zero for a constant feature.

Note that the normalisation is then applied to the **whole** array (`raw - fmean`).
That is correct: the *statistics* come only from training, but they are applied
everywhere, exactly as they would be in deployment.

## 4.5 The metrics

All metrics are computed on **raw case counts**, after inverting every transform. This
matters: an RMSE in log space is not comparable to anyone else's RMSE.

### RMSE — Root Mean Squared Error

```
RMSE = sqrt( mean( (pred − truth)² ) )
```

In cases per district-week. Because of the square, it is dominated by large errors,
which means it is dominated by outbreak weeks. That is arguably the right emphasis for
an epidemic — missing an outbreak is worse than being slightly off in a quiet week —
but you should know it is happening.

### MAE — Mean Absolute Error

```
MAE = mean( |pred − truth| )
```

Also in cases. Treats all errors proportionally. Reporting both RMSE and MAE tells the
reader how much of your error is concentrated in a few big misses: if RMSE ≫ MAE, it is.

### SMAPE — Symmetric Mean Absolute Percentage Error

```
SMAPE = mean( 2·|pred − truth| / (|pred| + |truth| + ε) ) × 100
```

The **primary percentage metric here**, because it is bounded at 200% and stays finite
when both values are zero — which happens in 9.74% of district-weeks.

### MAPE — zero-masked

```
MAPE = mean( |pred − truth| / truth ) × 100,   over weeks where truth ≥ 1
```

Kept only for comparability with papers that report it. Note the mask: without it, one
zero week gives infinity. Returns `nan` when no week qualifies, rather than a
misleading zero.

### 4.5.1 Peak week error — and a bug worth studying

Operationally the question is often not "how many cases" but **"when is the peak?"**.
Peak week error measures the gap, in weeks, between the predicted and observed peak.

The original implementation lived inside `score()` and did `argmax(axis=-1)`. Here is
why that was broken:

- On this project's `(n_windows, n_nodes, horizon)` arrays, `axis=-1` is **horizon** —
  so it took the argmax over 3 forecast steps.
- On a horizon-sliced 2-D array, `axis=-1` is **districts** — so it took the argmax
  over 25 districts.

It **never once looked along time.** It reported 0.0 for series shifted by any amount,
which looks like a perfect score.

The fix is a type signature that makes misuse impossible:

```python
if p.ndim != 2:
    raise ValueError(
        f"peak_week_error requires a 2D (n_weeks, n_nodes) array with time on "
        f"axis 0, got shape {p.shape}. Slice a single horizon first, e.g. "
        f"pred[:, :, h], or use peak_week_error_by_horizon()."
    )
```

The function now **requires** `(n_weeks, n_nodes)` and raises otherwise. You cannot
call it wrongly without an exception.

There is also a floor:

```python
PEAK_FLOOR = 10.0
qualifies = y.max(axis=0) >= min_peak
```

A district whose maximum is 3 cases has no outbreak to time; its argmax is noise. Those
districts are excluded rather than allowed to add random numbers to the average.

> **The general lesson.** A shape-agnostic function is right for elementwise metrics
> (RMSE, MAE, SMAPE all treat every element as an independent observation and genuinely
> do not care about axes). It is *wrong* for any metric that needs to know which axis is
> time. Separate them, and encode the requirement in the signature. A validation that
> proved the fix: persistence, scored this way, shows a peak error of about *h* weeks —
> exactly what a model that copies last week should show.

## 4.6 The baselines

A model is only interesting relative to something. This project uses six comparators,
and they are chosen to **bracket** the contribution from below.

### Persistence — the one that matters

```python
pred = np.repeat(raw[i - 1, :, cases_idx : cases_idx + 1], horizon, axis=1)
```

"Next 3 weeks all equal this week." Zero parameters, zero training, and — because
lag-1 autocorrelation is 0.68 — **very hard to beat**.

Under the frozen 8-fold protocol, persistence scores **RMSE 55.12**.

Any model that does not beat this has contributed nothing, no matter how sophisticated
it is. Reporting a GNN at RMSE 60 without reporting persistence at 55 is not an error
of arithmetic; it is an error of honesty.

### Seasonal naive

"This week last year" — `cases[t + h − 52]`. Dengue is seasonal, so this captures a
pattern persistence cannot express. It scores **RMSE 153.17** — much worse. Useful
precisely because it establishes that annual seasonality alone is not enough.

### Ridge regression

Linear regression with an L2 penalty on the coefficients, on the same 33-dimensional
window. Establishes **how much of the signal is linear** before any deep model gets
credit.

### Random forest and XGBoost

Tree ensembles over pooled district-weeks. These are the standard strong tabular
baselines and are what the reference literature reports as strong on this feature set.
Random forest scores **56.31**, XGBoost **57.41** — both competitive with everything
else, which is itself a finding.

### LSTM without a graph

A per-node sequence model that never sees the adjacency. This is the crucial control
for the project's central claim: it isolates **what the graph contributes**, as distinct
from what a deep model contributes. Scores **57.95**.

> **Design principle worth stealing.** Every baseline here shares the graph models'
> machinery *exactly* — the same `_build_fold` windows, the same residual target, the
> same `to_counts` inverse transform, the same `_rows_from` scoring. A baseline that
> differed in any of those would be measuring a different quantity. That is finding F6
> in a different costume.

## 4.7 Statistics — the part almost everyone skips

You have 24 numbers for model A and 24 for model B. A's mean is lower. **Is that real?**

### 4.7.1 The paired sign test

Because both models are evaluated on the *same* folds and seeds, the observations are
**paired**. Pairing is powerful: instead of comparing two noisy distributions, you ask
a much simpler question — *on how many of the 24 matched pairs did A beat B?*

Under the null hypothesis that they are equally good, each pair is a coin flip, so the
count of wins follows a binomial distribution:

```python
k = min(wins, n - wins)
p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
```

That is the exact two-sided binomial tail — no normal approximation, no assumption that
the differences are Gaussian. With n = 24 you need roughly **18 wins** for p < 0.05.

The pairing logic handles a subtlety:

```python
A = {(r["fold"], r["seed"]): float(r["rmse"]) for r in rows if r["label"] == a}
B = {(r["fold"], r["seed"]): float(r["rmse"]) for r in rows if r["label"] == b}
by_fold = {f: v for (f, _), v in B.items()}
...
vb = B.get((fold, seed), by_fold.get(fold))
```

It pairs on `(fold, seed)` where possible, and falls back to matching on fold alone
when the comparator is deterministic and has only one record per fold — which is
exactly the persistence case.

### 4.7.2 Statistical power — why 24 was not enough

The observed fold-to-fold spread here is enormous: SD around 39–44 against means around
55–60. The effects being chased are around 0.5–5 RMSE.

A **power analysis** on the project's own paired differences said that detecting those
effects at conventional significance would need **n ≈ 44–54** observations. The project
had 24.

This is why the results tables are full of `p=0.307`, `p=0.541`, `p=1.000`. Those are
not failures of the models. They are a **correctly reported inability to distinguish**
them at the available sample size. Reporting the mean alone would have implied a
confidence the data does not support.

> **Rule D8: negative results ship.** Report the failed contribution with its ablation
> row intact. Quietly dropping a row that did not work is the one thing here that is
> actually dishonest.

### 4.7.3 Aggregation sensitivity — a 1.53× trap

You have per-(fold, seed, horizon) errors. To report "the RMSE" you must aggregate. There
are at least three defensible ways:

1. **Pooled** — concatenate every prediction and compute one RMSE over all of them.
2. **Mean of per-horizon RMSEs** — compute RMSE for h=1, h=2, h=3, average the three.
3. **Mean of per-window RMSEs** — compute an RMSE for each test window, average them.

On this data these differ by **up to 1.53×**. Same predictions, same truth, three very
different numbers.

This is not academic. When reconciling against Weng et al. (2024), inspection of their
`evaluation.py` showed `batch_size = 1` with `rmse += RMSE(batch)` then `rmse /= n` —
which is aggregation method **3**. Persistence computed *their* way scores **38.46**,
and every model in their published Table I loses to it (STGAT 44.78, RF 84.66, LSTM
131.36).

So there was no contradiction with our finding at all. The apparent disagreement was
entirely an aggregation-convention difference.

> **Rule D2: always state your aggregation.** A number without its aggregation
> convention is not a result; it is a rumour. This project's rule requires the
> aggregation to be named next to every reported figure.

### 4.7.4 Rows are per (fold, seed, horizon) — never pre-aggregate

```python
def _rows_from(pred, truth, base):
    rows = []
    for h in range(pred.shape[-1]):
        s = score(pred[..., h], truth[..., h])
        rows.append({**base, "horizon": h + 1, ...})
    s = score(pred, truth)
    rows.append({**base, "horizon": 0, ...})   # 0 == pooled across horizons
    return rows
```

Every run writes one row per horizon **plus** a pooled row marked `horizon=0`. Nothing
is averaged before it reaches disk.

Why this is rule **D4**: the Phase-1 notebook averaged RMSE across seeds *inside* each
fold and stored only the mean. The seed dimension was destroyed before it was ever
written down — which is why no Phase-2 result has a standard deviation and why no
significance test on them is possible. That is review finding **F7**, and it is
unrecoverable without re-running everything.

**Write raw. Aggregate at reporting time.** Then you can recompute any aggregation,
drop a bad seed, or run a different test, without touching a GPU.

---

# Part V — The Target Parameterisation

This is the single most consequential modelling decision in the project. It moved RMSE
from ~66 to ~45 — a bigger effect than every architectural choice combined. It is also
the source of the project's most subtle failure mode.

Recorded formally as **ADR-0001** (`docs/decisions/0001-baseline-training-refinements.md`).

## 5.1 log1p — taming the tail

The problem: median 13 cases, maximum 2,631. Squared error on raw counts means one
outbreak week at 2,000 contributes as much as 23,000 quiet weeks at 13. The model
optimises almost entirely for the tail and predicts a bland near-mean everywhere else.

The fix: train on `log1p(cases) = log(1 + cases)`.

Why `log1p` and not `log`: 9.74% of observations are zero, and `log(0) = −∞`. Adding 1
first maps 0 to 0 and keeps everything finite. The inverse is `expm1(x) = e^x − 1`,
which maps 0 back to 0 exactly. NumPy and PyTorch both provide these as single
functions, and they are more numerically accurate than writing `log(1+x)` yourself for
small x.

In log space, the ratio between a quiet week and an outbreak is compressed from 200× to
about 5×, so both contribute meaningfully to the loss.

The inverse transform:

```python
def to_counts(self, normalised):
    """Invert normalisation and log1p, differentiably."""
    return torch.expm1(torch.clamp(normalised * self.target_std + self.target_mean, 0, 12))
```

Three things happen here, in order:

1. **Undo z-normalisation**: `x * std + mean`.
2. **Clamp to [0, 12]** in log space. `e^12 ≈ 162,754` cases is an absurd forecast; the
   clamp suppresses occasional hallucinated spikes. It is differentiable inside the
   range, so it does not break gradients where it matters.
3. **Undo log1p** with `expm1`.

The clamp at 0 also has a consequence that is easy to miss and is documented in the
code: predicted counts are **non-negative by construction**. So the non-negativity
penalty in the loss (`L_cons`) is *identically zero under every configuration*. The
`10.0 · L_cons` term in the published equation contributes literally nothing. It is
kept only so the objective matches the specification — but any paper claiming it as a
contribution would be claiming an inert term.

## 5.2 Residual over persistence — the big one

Since lag-1 autocorrelation is 0.68, persistence is already a strong forecast. A model
predicting absolute counts must **re-learn that autocorrelation from scratch** before it
can add anything. That is wasted capacity, and on 459 weeks of data you cannot afford it.

So instead, the model predicts the **correction to persistence**:

```
prediction = persistence + model_output
```

Training target:

```python
target = fold.y_train[i] - fold.p_train[i] if cfg.residual else fold.y_train[i]
```

where `p` is the persistence anchor — last week's value, repeated across the horizon,
in the same normalised log space:

```python
ps.append(np.repeat(tgt[i - 1].reshape(n_nodes, 1), cfg.horizon, axis=1))
```

At evaluation the correction is added back:

```python
if cfg.residual:
    out = gamma * out + p[i]
```

Now the model only has to learn *what is different about this week*, which is a far
smaller and better-posed problem.

### 5.2.1 The measured evidence

An ablation, run under the frozen protocol:

| Setting | RMSE |
|---|---|
| residual + log1p | **85.32** |
| absolute + log1p | 187.57 |
| absolute, no log | 70,318 |

And from the earlier Phase-1 study:

| Setting | RMSE | MAE |
|---|---|---|
| Plain GCN (absolute scale) | 66.2 | 31.6 |
| + log1p only | 67.1 | 27.0 |
| + residual only | 45.0 | 16.8 |
| residual + log | 45.3 | 15.8 |

Read the second table carefully — it contains a surprise. **log1p alone makes RMSE
worse** (66.2 → 67.1). It only helps once combined with the residual, and even then its
contribution is to MAE rather than RMSE.

The residual is doing essentially all of the work. This is the kind of thing an
ablation exists to reveal, and the kind of thing that gets misattributed when you change
two things at once and report the combined result.

## 5.3 Residual shrinkage

If the model predicts a correction, you can scale that correction:

```
prediction = persistence + γ · model_output
```

- γ = 0 → exactly persistence
- γ = 1 → the unshrunk model
- 0 < γ < 1 → a blend

γ is chosen on the **validation** fold by grid search:

```python
def select_shrinkage(model, fold, cfg, device):
    if not cfg.shrink or not cfg.residual:
        return 1.0
    best_gamma, best_rmse = 1.0, float("inf")
    for gamma in cfg.shrink_grid:
        pred, truth = _evaluate(model, fold.x_val, fold.y_val, fold.p_val, fold, cfg, device, gamma)
        rmse = score(pred, truth)["RMSE"]
        if rmse < best_rmse:
            best_gamma, best_rmse = gamma, rmse
    return best_gamma
```

The comment in the source states the principle exactly: *"γ is chosen on the VALIDATION
fold and then applied to test — selecting it on test would be choosing the answer."*
The validation fold is the same one early stopping uses, so no additional data is
consumed and the test fold stays untouched.

## 5.4 ⚠ The residual collapse — the trap at the heart of this design

Here is the failure mode that the residual parameterisation creates, and which this
project only caught by looking for it deliberately.

**If the model outputs zero, it *is* persistence, and it inherits persistence's score.**

Persistence scores RMSE 55.12 here — competitive with every model in the study. So
outputting zero is a *safe, competitive local minimum*. A model that finds it will:

- have a low RMSE
- possibly the **lowest** RMSE in your table
- and be forecasting **nothing at all**

### 5.4.1 How it was caught

In the architecture study, STGAT came top:

| arm | RMSE mean | vs persistence |
|---|---|---|
| persistence | 55.12 | — |
| **STGAT** | **54.95** | 15/24, p=0.307 |
| A3TGCN | 56.19 | 11/24, p=0.839 |
| GCN + gated TCN | 56.39 | 12/24, p=1.000 |
| GAT | 59.99 | 10/24, p=0.541 |
| GCN (control) | 60.08 | 9/24, p=0.307 |

The lowest mean, and the only arm below persistence. It looks like the headline result.

The tell was **not** in the mean. It was in the spread of the *paired differences*:

| arm | mean paired difference vs persistence | SD |
|---|---|---|
| X_gcn | +4.96 | 11.97 |
| X_gcn_gtcn | +1.27 | 9.50 |
| X_gat | +4.87 | 16.97 |
| **X_stgat** | **−0.16** | **0.70** |
| X_a3tgcn | +1.08 | 9.57 |

Every other arm deviates from persistence by an SD of 9.5–17. STGAT deviates by
**0.70**. It is tracking persistence to within less than one RMSE point on every single
fold. That is not a model that agrees with persistence on average — it is a model that
*is* persistence.

### 5.4.2 Measuring it directly

The diagnostic (`scripts/diagnose_residual.py`) measures two quantities:

```
move   = mean |prediction − persistence|      what the model actually adds
needed = mean |truth      − persistence|      what it would have to add
ratio  = move / needed
```

Results, seed 0, three folds:

| arm | fold 0 | fold 4 | fold 7 |
|---|---|---|---|
| **STGAT** | **0.053** | **0.023** | **0.042** |
| GAT | 1.127 | 0.268 | 0.059 |
| GCN + gated TCN | 0.524 | 0.143 | 0.128 |
| A3TGCN | 0.393 | 0.288 | 0.087 |
| GCN (control) | 0.348 | 0.407 | 0.431 |

STGAT moves **2–5%** of the required distance. Confirmed: it collapsed.

Its 748 seconds of training bought a **36× cost increase** over the control for a copy
of a baseline that costs nothing.

### 5.4.3 The general lesson, and the fix

> **RMSE cannot distinguish a forecast from a copy of the baseline.** No accuracy
> metric can. If your model is parameterised as a correction to a strong baseline, you
> must *separately* verify that it is making corrections.

The fix was to make this a standing metric. `resid_ratio` is now computed on every
result row, in `run_single`:

```python
persist_pred, _ = _persistence_counts(raw, cfg, test_ids)
needed = float(np.mean(np.abs(truth - persist_pred)))
moved  = float(np.mean(np.abs(pred  - persist_pred)))
resid_ratio = moved / needed if needed else float("nan")
```

Now the failure mode is visible in the CSV rather than requiring a bespoke
investigation. A value near 0 means the arm is not forecasting. A value near 1 means it
is making corrections of the right magnitude — which says nothing about whether they
point the *right way*, only that it is making them. Report it beside RMSE, never
instead of it.

> **This also sharpens the reading of published work.** A published RMSE for STGAT is
> not by itself evidence that the model forecasts. No paper reviewed in this project
> reports a statistic that could separate the two cases. That is not an accusation
> against any specific paper — it is an observation that the field's standard reporting
> is not sufficient to rule this out.

## 5.5 Capacity matching — the other attribution trap

The first architecture table compared arms at a fixed hidden width of 64. Measured
parameter counts on this dataset:

| arm | parameters |
|---|---|
| GCN (control) | 7,032 |
| GCN + gated TCN | 11,960 |
| A3TGCN | 23,865 |
| GAT (8 heads) | 52,408 |
| STGAT | 87,992 |

GAT carries **7.5×** the control's parameters; STGAT **12.5×**. On 25 nodes and 459
weeks, that gap is on its own a plausible explanation for any result in either
direction. So "GAT vs GCN" at default width varied **architecture and capacity
together** — precisely the confound of finding F6.

The fix is matched pairs:

| arm | parameters | matched against |
|---|---|---|
| GAT 1-head | 7,160 | GCN control, 7,032 — a 1.8% match |
| GAT 2-head | 13,624 | GCN h96, 13,368 — a 1.9% match |
| GCN h128 | 21,752 | A3TGCN, 23,865 — an 8.9% match |

With capacity held fixed the picture does not improve for attention. GAT 1-head scores
**68.91** against the control's **60.08** — attention *loses* to uniform propagation
when width is held constant. And GAT 2-head is significantly **worse** than persistence
(5/24 wins, p = 0.007).

> **The principle.** When comparing architectures, hold parameter count roughly fixed,
> or you are comparing capacity. State the counts in the table so a reader can check.

---

# Part VI — The Code, Module by Module

## 6.0 Repository layout

```
src/dengue_gnn/          the library — everything that produces a number
  __init__.py
  models.py              network architectures
  experiment.py          the evaluation harness (the core)
  metrics.py             scoring functions
  losses.py              spatial regularisation
  mechanistic.py         epidemiological constraints
  augment.py             data augmentation, including the GAN
  baselines.py           non-graph comparators
  provenance.py          run manifests
  results_logger.py      CSV writing

scripts/                 orchestration — runnable entry points
  run_phase3.py          the main experiment runner
  make_tables.py         results → markdown/LaTeX tables
  compare_arch.py        architecture comparison table
  diagnose_residual.py   the collapse diagnostic
  index_runs.py          cross-run index
  measure_loss_scale.py  loss magnitude measurement
  kaggle_run.py          remote execution

tests/                   113 tests
docs/                    decisions, reviews, experiment log
results/                 CSVs + manifests + generated tables
paper/                   LaTeX source
```

> **Rule D3, and it is the most expensive lesson in the project.** Code that produces a
> number lives in `src/`, never in a git-ignored `scratch/` directory. *Every* Phase-2
> result was unreproducible because the code that made it was never committed. It had
> to be reconstructed from the paper's equations and verified bit-for-bit against the
> committed CSVs. Do not repeat this.

## 6.1 `models.py`

### `row_normalize(adj, eps=1e-8)`

```python
def row_normalize(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(eps)
```

`keepdim=True` preserves the shape as *(N, 1)* so broadcasting divides each row by its
own sum. Without it you would get shape *(N,)* and NumPy/PyTorch broadcasting rules
would divide each *column* instead — a silent transposition bug.

### `build_fixed_adjacency(adj_list, node_order, self_loops=True, normalize=True)`

Converts the JSON adjacency list into an *(N, N)* tensor.

```python
index = {name: i for i, name in enumerate(node_order)}
adj = torch.zeros(n, n, dtype=torch.float32)
for district, neighbours in adj_list.items():
    if district not in index:
        raise ValueError(f"district {district!r} missing from node_order")
    i = index[district]
    for neighbour in neighbours:
        if neighbour not in index:
            raise ValueError(f"neighbour {neighbour!r} missing from node_order")
        adj[i, index[neighbour]] = 1.0
if self_loops:
    adj = adj + torch.eye(n)
if normalize:
    adj = row_normalize(adj)
```

**`node_order` is passed explicitly rather than inferred from the dict's key order.**
This is load-bearing. Axis 1 of the feature array follows a particular district order;
if the adjacency is built in a different order, the graph is silently scrambled — every
district gets another district's neighbours. Nothing raises. Your model just quietly
learns from a randomised map. The explicit parameter plus the two `ValueError`s make
the mismatch loud.

### `AdaptiveAdjacency`

Covered in §3.4. Two `nn.Parameter` embedding matrices; forward returns
`softmax(relu(e1 @ e2.T))`.

### `GatedTCN`

Covered in §3.6.2. Note the constructor stores `self.pad = (kernel - 1) * dilation` so
the causal left-padding is computed once.

### `GraphAttention`

Covered in §3.5. Two implementation details worth calling out.

**The attention vector is split.** The paper writes a single vector `a` applied to the
concatenation `[h_i ; h_j]`. Splitting it into `att_src` and `att_dst` is
mathematically identical (`a·[h_i;h_j] = a_src·h_i + a_dst·h_j`) but avoids
materialising an *N × N × 2F* concatenation tensor. For N=25 that is not critical; for
larger graphs it is the difference between running and running out of memory.

**Xavier initialisation:**

```python
nn.init.xavier_uniform_(self.proj.weight)
nn.init.xavier_uniform_(self.att_src)
nn.init.xavier_uniform_(self.att_dst)
```

Xavier (Glorot) initialisation scales the initial random weights by the layer width so
that activation variance is preserved through the network. Default PyTorch
initialisation is usually fine, but attention logits feed a softmax, which is sensitive
to scale — too large and the softmax saturates into a hard argmax with vanishing
gradients.

### `RecurrentTemporal`

Covered in §3.6.3. Validates `kind` in the constructor:

```python
if kind not in ("gru", "lstm"):
    raise ValueError(f"kind must be 'gru' or 'lstm', got {kind!r}")
```

Fail at construction, not three hours into a run.

### `AdaptiveGCN` — the main model

The constructor validates aggressively, and each check prevents a specific silent
failure:

```python
if adj_fixed.shape != (n_nodes, n_nodes):
    raise ValueError(...)                      # wrong graph entirely
if temporal not in valid_temporal:
    raise ValueError(...)                      # typo → silently no temporal operator
if spatial not in ("gcn", "gat"):
    raise ValueError(...)
if n_feat * window != in_dim:
    raise ValueError(...)                      # reshape would scramble the window
```

The last one deserves attention. When a temporal operator is active, the forward pass
reshapes the flat input back into a sequence:

```python
seq = x.reshape(b * n, self.n_feat, self.window)
```

If `n_feat * window != in_dim`, this reshape either raises or — worse — succeeds with
wrong dimensions and silently interleaves features with timesteps. Checking at
construction turns a subtle numerical disaster into an immediate, readable error.

**The forward pass:**

```python
def forward(self, x):
    if x.dim() != 3 or x.shape[1] != self.n_nodes:
        raise ValueError(...)

    adj = self.blended_adjacency()

    if self.tcn is not None:
        b, n, _ = x.shape
        seq = x.reshape(b * n, self.n_feat, self.window)
        x = self.tcn(seq).reshape(b, n, -1)

    if self.spatial == "gat":
        h = F.relu(self.gat1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.gat2(h, adj))
    else:
        h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w1(x)))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(torch.einsum("ij,bjf->bif", adj, self.w2(h)))
    return self.head(h)
```

The `reshape(b * n, ...)` folds nodes into the batch axis, so the temporal operator is
applied independently to every district. That is correct: the temporal operator models
*within-district* dynamics, and the graph layers handle *across-district* mixing. Then
`.reshape(b, n, -1)` unfolds them again.

`training=self.training` on the functional dropout is what makes `model.eval()` work —
`self.training` is the flag `train()`/`eval()` toggle.

## 6.2 `experiment.py` — the harness

The most important file. ~585 lines.

### `Config` — the dataclass

Every knob in one place, with defaults, grouped by purpose:

```python
@dataclass
class Config:
    # data / protocol -- do not change without re-running every ablation row
    window: int = 3
    horizon: int = 3
    cases_idx: int = 5
    origins: tuple[float, ...] = (0.55, 0.70, 0.85)
    test_frac: float = 0.15
    val_weeks: int = 30
    seeds: tuple[int, ...] = (0, 1, 2)
    # target handling
    residual: bool = True
    log_transform: bool = True
    # model
    hidden: int = 64
    temporal: str = "none"
    spatial: str = "gcn"
    ...
```

Why a dataclass rather than a pile of function arguments:

1. **One object defines a run completely.** It can be serialised into every result row,
   so a CSV row is self-describing.
2. `asdict()` gives you a manifest for free.
3. Defaults live in one place, so two runners cannot drift apart.

`as_row()` flattens tuples for CSV:

```python
def as_row(self):
    d = asdict(self)
    d.pop("extra", None)
    d["origins"] = ",".join(str(o) for o in self.origins)
    d["seeds"]   = ",".join(str(s) for s in self.seeds)
    return d
```

### `FoldData` — one fold's tensors

Nine tensors (`x`, `y`, `p` for train/val/test), plus `target_mean`, `target_std`, and
`test_ids`. Bundling them means functions take one argument rather than twelve, and it
is impossible to pass the validation `x` with the training `y`.

Its one method is `to_counts`, the inverse transform, covered in §5.1.

### `load_dataset(npy_path, adj_path, self_loops=True)`

```python
raw = np.nan_to_num(np.load(npy_path, allow_pickle=True)).astype(np.float32)
with open(adj_path) as fh:
    adj_list = json.load(fh)

names = sorted(adj_list)
if raw.shape[1] != len(names):
    raise ValueError(f"array has {raw.shape[1]} nodes but adjacency has {len(names)} districts")
adj = build_fixed_adjacency(adj_list, names, self_loops=self_loops, normalize=True)
return raw, adj, names
```

- `np.nan_to_num` replaces NaN with 0. Satellite data has gaps; a single NaN propagates
  through the entire network and turns every gradient into NaN.
- `float32` rather than float64 — half the memory, and enough precision for this.
- `sorted(adj_list)` gives alphabetical district order, matching the Phase-1 notebook.
  The docstring is explicit that axis 1 is *assumed* to follow the same order and that a
  mismatch scrambles the graph silently. The node-count check catches the gross case.

### `_build_fold(raw, cfg, train_ids, eval_ids)`

Where windowing and normalisation happen. This is the function to understand most
carefully.

```python
n_nodes, n_feat = raw.shape[1], raw.shape[2]
train_weeks = raw[: train_ids[-1]]

fmean = train_weeks.reshape(-1, n_feat).mean(0)
fstd  = train_weeks.reshape(-1, n_feat).std(0) + 1e-6
feat  = (raw - fmean) / fstd
```

Feature normalisation from training statistics only (§4.4). `reshape(-1, n_feat)`
flattens time and space together so the statistics are per-feature across all
district-weeks.

```python
cases = raw[..., cfg.cases_idx]
if cfg.log_transform:
    base  = np.log1p(cases)
    tbase = np.log1p(train_weeks[..., cfg.cases_idx])
    tmean, tstd = float(tbase.mean()), float(tbase.std() + 1e-6)
else:
    base  = cases
    tmean = float(train_weeks[..., cfg.cases_idx].mean())
    tstd  = float(train_weeks[..., cfg.cases_idx].std() + 1e-6)
tgt = (base - tmean) / tstd
```

The target is normalised **separately** from the features, with its own mean and SD.
Those two scalars are carried in `FoldData` because `to_counts` needs them to invert.

Then the window builder:

```python
def make(ids):
    xs, ys, ps = [], [], []
    for i in ids:
        xi = np.transpose(feat[i - cfg.window : i], (1, 2, 0)).reshape(
            n_nodes, n_feat * cfg.window
        )
        xs.append(xi)
        ys.append(tgt[i : i + cfg.horizon].T)
        ps.append(np.repeat(tgt[i - 1].reshape(n_nodes, 1), cfg.horizon, axis=1))
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float)
    return t(xs), t(ys), t(ps)
```

For each origin `i`:

**Input `xi`.** `feat[i-3 : i]` is *(3 weeks, 25 nodes, 11 features)*. The transpose
`(1, 2, 0)` reorders to *(25 nodes, 11 features, 3 weeks)*, and the reshape flattens the
last two into *(25, 33)*.

> **This layout is load-bearing and appears in four places.** It is **feature-major**:
> the 33 numbers are ordered as *feature 0 at weeks 1,2,3; feature 1 at weeks 1,2,3;
> …*. Any code that reshapes this flat vector back into a sequence must use
> `reshape(n_feat, window)` — which is exactly what `AdaptiveGCN.forward`,
> `_LSTM.forward` in `baselines.py`, and `window_warp` in `augment.py` all do. Use
> `(window, n_feat)` instead and you get a silently scrambled window that trains
> without complaint and predicts badly. Notice that all four sites carry a comment
> pointing back to this transpose. That is deliberate.

**Target `ys`.** `tgt[i : i+3]` is *(3 weeks, 25 nodes)*; `.T` makes it *(25, 3)* —
nodes first, horizon second, matching the model's output.

**Persistence anchor `ps`.** `tgt[i-1]` is last observed week, *(25,)*; reshaped to
*(25, 1)* and repeated 3 times to *(25, 3)*. This is the value the residual is measured
against.

Note `i - 1` is the week *before* the forecast origin — the last week actually
observed. Off-by-one here would leak the first forecast week into the anchor.

### `_evaluate(model, x, y, p, fold, cfg, device, gamma=1.0)`

```python
model.eval()
preds, truths = [], []
with torch.no_grad():
    for i in range(x.shape[0]):
        out = model(x[i].unsqueeze(0).to(device)).squeeze(0).cpu()
        if cfg.residual:
            out = gamma * out + p[i]
        preds.append(fold.to_counts(out))
        truths.append(fold.to_counts(y[i]))
return torch.stack(preds).numpy(), torch.stack(truths).numpy()
```

- `model.eval()` disables dropout.
- `torch.no_grad()` stops PyTorch building the computation graph — faster and much
  lighter on memory, since we are not going to backpropagate.
- `unsqueeze(0)` adds a batch axis of 1; `squeeze(0)` removes it.
- `gamma * out + p[i]` applies shrinkage then adds the persistence anchor.
- `to_counts` on **both** prediction and truth, so metrics are computed on raw counts.

### `train_fold(fold, cfg, adj_fixed, device)`

The training loop. Builds the model from config, then:

```python
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
mse = nn.MSELoss()

best, best_state, wait = float("inf"), None, 0
for epoch in range(cfg.epochs):
    model.train()
    for i in torch.randperm(fold.x_train.shape[0]):
        opt.zero_grad()
        out = model(fold.x_train[i].unsqueeze(0).to(device))
        target = fold.y_train[i] - fold.p_train[i] if cfg.residual else fold.y_train[i]
        loss = mse(out.squeeze(0), target.to(device))
        ...
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    pred, truth = _evaluate(model, fold.x_val, fold.y_val, fold.p_val, fold, cfg, device)
    val_rmse = score(pred, truth)["RMSE"]
    ...
```

**`torch.randperm`** shuffles the *order of training windows within an epoch*. This is
not temporal leakage — each window is already a self-contained (past → future) pair, and
shuffling their presentation order just decorrelates consecutive gradient updates.
Shuffling *within* a window would be leakage; shuffling *between* windows is standard.

**Batch size 1.** Each window is a separate optimiser step. Unusual, but inherited from
the Phase-1 notebook and kept deliberately — changing it would change results and break
comparability with every earlier row (D1). It is also less costly than it sounds here:
each "sample" is a full 25-node graph.

### The regularisation block

```python
if cfg.lambda_phys > 0.0 or cfg.lambda_mech > 0.0:
    normalised = out.squeeze(0)
    if cfg.residual:
        normalised = normalised + fold.p_train[i].to(device)

    scale = (
        curriculum_weight(epoch, cfg.epochs, 1.0, cfg.curriculum)
        if cfg.curriculum > 0
        else 1.0
    )
```

Crucial detail: the regularisers act on **predictions**, not residuals, so the
persistence anchor must be added back first. See §6.4 for why this distinction is not
cosmetic.

### `select_shrinkage`, `_regularisation_target`

Covered in §5.3 and §6.4.

### `run_single(raw, adj_fixed, cfg, fold_idx, seed, device=None)`

The unit of work. One (fold, seed) → a list of result rows. It:

1. Splits the fold.
2. Builds train/val and train/test tensors — note `_build_fold` is called **twice**,
   both times with `train_ids` first, so validation and test are normalised with the
   *same* training statistics.
3. Optionally augments the training set.
4. Seeds the RNGs.
5. Trains, timing it.
6. Selects shrinkage on validation.
7. Evaluates on test, timing inference.
8. Computes `resid_ratio`.
9. Assembles a `base` dict and calls `_rows_from`.

The `base` dict carries everything needed to interpret the row later:

```python
base = {
    "label": cfg.label, "fold": fold_idx + 1, "origin": origin, "seed": seed,
    "model": "AdaptiveGCN" if cfg.use_adaptive else "DenseGCN",
    "lambda_phys": cfg.lambda_phys,
    "gate_sigma": model.gate_value(),
    "shrink_gamma": gamma,
    "resid_ratio": round(resid_ratio, 4),
    "temporal": cfg.temporal, "spatial": cfg.spatial,
    "window": cfg.window, "hidden": cfg.hidden, "lr": cfg.lr,
    "augment": cfg.augment,
    "n_train_windows": int(xtr.shape[0]), "n_real_windows": int(n_real),
    "n_test_weeks": len(test_ids),
    "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "train_seconds": round(train_seconds, 3),
    "infer_ms_per_window": round(1000 * infer_seconds / max(len(test_ids), 1), 4),
}
```

Timing and parameter counts are recorded **during** the run, because the paper needs a
Computational Analysis section and measuring it afterwards means running everything
twice.

`n_train_windows` vs `n_real_windows` distinguishes augmented from real data, so an
augmentation row is self-describing.

### `rolling_origin` and `run_single` — one definition of a run

```python
for fold_idx in range(len(cfg.origins)):
    for seed in cfg.seeds:
        fold_rows = run_single(raw, adj_fixed, cfg, fold_idx, seed, device)
```

`rolling_origin` is the sequential runner and it **delegates to `run_single`**, which is
also what the parallel runner calls. The comment explains why: *"Two copies of split
arithmetic drift by a window and silently stop being the same protocol."* If you
maintain a sequential and a parallel path separately, they will diverge, and you will
not notice.

## 6.3 `metrics.py`

Covered in §4.5. The architectural point is the separation:

- `score()` and `metrics()` are **shape-agnostic** — correct for elementwise metrics.
- `peak_week_error()` **requires** `(n_weeks, n_nodes)` and raises otherwise — because
  it needs to know which axis is time.

`_check_pair` is shared validation:

```python
def _check_pair(p, y):
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs truth {y.shape}")
    if p.size == 0:
        raise ValueError("cannot score empty arrays")
```

Scoring empty arrays returns `nan` in NumPy rather than raising, and a `nan` in your
results table looks like a missing value rather than a bug.

## 6.4 `losses.py` — spatial regularisation

The objective in the Phase-2 paper:

```
L_total = L_data + λ · ( L_smooth + cons_weight · L_cons )
```

### `smoothness_loss(pred_counts, adj)`

A graph-Laplacian quadratic form — penalise neighbouring districts disagreeing:

```
L_smooth = Σ_ij A_ij (ŷ_i − ŷ_j)²  /  (B · H · ‖A‖₁)
```

```python
diff = pred_counts.unsqueeze(2) - pred_counts.unsqueeze(1)
weighted = adj.unsqueeze(0).unsqueeze(-1) * diff.pow(2)
batch, _, _, horizon = weighted.shape
denom = batch * horizon * adj.abs().sum().clamp_min(1e-8)
return weighted.sum() / denom
```

`unsqueeze(2) - unsqueeze(1)` broadcasts *(B, N, 1, H) − (B, 1, N, H)* into
*(B, N, N, H)* — all pairwise differences in one operation. Normalising by `‖A‖₁` keeps
the term comparable across adjacency matrices with different total edge mass, which
matters because the blended adjacency changes during training.

The intent is to suppress a real failure mode: a GNN forecasting a large outbreak in
one district and nothing next door, purely as an artefact of unconstrained message
passing.

### `nonnegativity_loss(pred_counts)`

```python
return torch.relu(-pred_counts).pow(2).mean()
```

Penalises negative predicted counts. As established in §5.1, **this term is identically
zero** because `to_counts` clamps before `expm1`. It is kept only to match the published
specification.

### ⚠ Which space these act in

Both terms are statements about **case counts**, not residuals. Every function takes
`pred_counts`.

> This is not stylistic. `L_cons` penalises negative predictions on the grounds that a
> district cannot have negative cases. Applied to the raw network output, a negative
> value does not mean "negative cases" — it means "**fewer cases than last week**",
> which is a correct and frequent forecast. Penalising it would systematically bias the
> model against ever predicting a decline. On a seasonal disease that is a serious
> distortion.

The original implementation was lost, so which space it used cannot be determined. The
API forces the correct one by naming the parameter `pred_counts`.

### `_regularisation_target` — and a measurement that changed the design

Whether to regularise on counts or `log1p(counts)`:

| space | L_smooth magnitude | vs data loss (~0.27) |
|---|---|---|
| count | ~4,800 | ~17,700× |
| log | order 1 | ~1× |

On raw counts, at the paper's *smallest* non-zero λ (0.01), the regulariser is **99.4%
of the total loss** and the model stops fitting data entirely. Since the published λ
sweep moved RMSE by only 0.07, the lost original **cannot** have been applying the
constraint on that scale — corroborating review finding F5.

`log` is the default for two reasons: magnitudes land within an order of magnitude of
the data loss so λ behaves like a weight rather than a switch; and on a target with
median 13 and maximum 2,631, smoothness on raw counts is dominated entirely by the
largest districts, whereas in log space it penalises *relative* disagreement between
neighbours — which is the property the constraint actually appeals to.

> **Generalisable lesson (recorded as LL-019).** Before choosing a regularisation
> weight, **measure the magnitude of each loss term**. A λ grid chosen by intuition can
> sit two orders of magnitude away from the usable range, and you will conclude the
> constraint does not work when you never actually tested it.
> `scripts/measure_loss_scale.py` exists solely to do this.

### The zero-λ shortcut

```python
if lambda_phys == 0.0:
    return pred_counts.sum() * 0.0
```

Returns a differentiable zero *without* building the *(B, N, N, H)* pairwise tensor. So
the λ=0 ablation row costs nothing extra **and is exactly the unregularised model**
rather than an approximation. Returning a plain `0.0` would break the autograd graph;
`pred_counts.sum() * 0.0` keeps it connected while contributing nothing.

### A naming honesty note

The proposal committed to a physics-informed loss derived from the SEIR-SEI
compartmental model. What is implemented is a graph-Laplacian smoothness penalty plus a
non-negativity constraint. **That is spatial regularisation, not mechanistic
epidemiology** — no compartments, no transmission dynamics, no ODE residual. The module
is named `losses.py` with functions named `spatial_regularisation`, not
`physics_loss`. Naming code for what it does rather than what you hoped it would do is
how you avoid your own paper overclaiming.

## 6.5 `mechanistic.py` — constraints that only use observables

### Why not a full SEIR-SEI residual

The SEIR-SEI host-vector model has **seven compartments**: susceptible/exposed/
infectious/recovered humans and susceptible/exposed/infectious mosquitoes. Exactly
**one** of them — infectious humans, and only as under-ascertained *reported cases* —
is observed here.

A hard ODE residual would ask the network to infer **150 latent weekly trajectories**
(6 unobserved compartments × 25 districts) plus nine rate constants, from 25 noisy
series. That system is **structurally under-determined**: many latent configurations
reproduce the same observed cases, and nothing in the loss prefers the correct one.

So the constraints instead act on quantities computable from predictions alone, deriving
their form from the **generation interval** — the mechanistic quantity that survives when
compartments are unobservable.

### `implied_log_growth(pred_counts, last_observed)`

```python
log_pred = torch.log1p(pred_counts.clamp_min(0))
anchor   = torch.log1p(last_observed.clamp_min(0)).unsqueeze(-1)
series   = torch.cat([anchor, log_pred], dim=-1)
return series[..., 1:] - series[..., :-1]
```

Prepends the last observed value, then takes first differences — the week-on-week log
growth rate implied by the forecast. Prepending the anchor is what makes the *first*
horizon step's growth well defined.

### `growth_band_loss`

```python
g = implied_log_growth(pred_counts, last_observed)
return torch.relu(g.abs() - max_growth).pow(2).mean()
```

A hinge: zero inside the plausible band, quadratic outside. `MAX_WEEKLY_LOG_GROWTH =
0.70` is derived, not guessed:

> Dengue's generation interval (intrinsic incubation + infectious period + extrinsic
> incubation) is roughly 3 weeks. A per-generation reproduction number *R* implies weekly
> log growth ≈ ln(R)/3. *R = 8* is already extreme for dengue, giving ln(8)/3 = 0.69.

Unlike spatial smoothness this does **not** push toward a uniform solution — it
constrains only the tails, so it has no incentive to flatten the graph structure the
adaptive component is learning.

### `growth_smoothness_loss`

Penalises the second difference of `log(1+cases)`. The effective reproduction number
moves on the timescale of the generation interval, so week-to-week growth should not
jump. This is the renewal-equation intuition without instantiating a renewal kernel.

### `curriculum_weight(epoch, total_epochs, target, warmup=0.5)`

```python
frac = min(1.0, epoch / max(1.0, warmup * total_epochs))
return target * frac
```

Krishnapriyan et al. find that physics-informed network failures are often
**optimisation** failures: a soft constraint at full strength from step one deforms the
loss landscape so badly that gradient descent cannot navigate it. Ramping the
coefficient recovers one to two orders of magnitude of error.

Since Stage 2b used a fixed weight throughout, this is the cheapest available test of
whether *that* — rather than the constraint itself — is what failed.

### `adaptive_weight(...)` — gradient-norm balancing

After Wang, Teng & Perdikaris. Rather than fixing λ by hand, estimate it from gradient
statistics:

```
λ̂ = max|∇L_data| / mean|∇L_constraint|
λ ← (1−α)·λ + α·λ̂          with α = 0.1
```

The diagnosis: the two loss terms produce gradients of wildly different magnitude, so
one dominates and the other is ignored — precisely what was measured here, where the
spatial penalty exceeded the data loss by four orders of magnitude.

```python
if mean_cons <= eps or max_data <= eps:
    return current  # inert constraint, or a converged data term
```

That guard matters: without it, an inert constraint (gradient ≈ 0) makes the ratio blow
up to infinity and destroys training. Returning `current` unchanged means an inert term
cannot drive the weight anywhere.

Smoothing matters too — the raw ratio is noisy at batch size 1 and an unsmoothed weight
oscillates.

> **Rule D13 — never delegate the novelty.** Every constant in this module is marked
> `OWNER:` and must be confirmed by a human author before it reaches the paper. A subtly
> wrong ODE that trains fine is far worse than one that crashes.

## 6.6 `augment.py` — Contribution (b)

459 weeks × 25 districts is a small corpus for a deep model. Can we synthesise more
training pairs?

### `jitter(x, sigma=0.05)`

```python
return x + torch.randn_like(x) * sigma
```

Additive Gaussian noise on normalised (unit-variance) features. The cheapest useful
augmentation.

### `window_warp(x, window, n_feat, scale=0.2)`

```python
seq = x.reshape(n_nodes, n_feat, window)
rate = float(1.0 + (torch.rand(1).item() * 2 - 1) * scale)
warped_len = max(2, round(window * rate))
stretched = nn.functional.interpolate(seq, size=warped_len, mode="linear", align_corners=True)
back = nn.functional.interpolate(stretched, size=window, mode="linear", align_corners=True)
return back.reshape(n_nodes, n_feat * window)
```

Resample the time axis at a random rate in 1 ± 0.2, then resample back. The result is a
window whose dynamics ran slightly faster or slower — plausible for an epidemic whose
speed varies with climate. Note the `reshape(n_nodes, n_feat, window)`: the
feature-major layout again.

### `TimeSeriesWGAN`

A **GAN** (Generative Adversarial Network) pits two networks against each other: a
**generator** producing fake samples and a **critic** scoring them. The generator learns
to fool the critic; the critic learns to tell real from fake. At equilibrium the
generator produces realistic samples.

Three specific choices here:

**Conditional (RCGAN-style) rather than unconditional (TimeGAN-style).** The generator
receives the observed covariate window and emits the corresponding target sequence, so a
synthetic sample is a plausible *continuation of real conditions* rather than a free
invention. At this sample size an unconditional generator has 459 sequences to learn
from and will memorise or collapse.

**Wasserstein with gradient penalty (WGAN-GP) rather than the original GAN objective.**
Mode collapse — the generator producing one output regardless of input — is the named
risk for this dataset in the project's risk register, and WGAN-GP is the standard
mitigation. Note the critic has **no sigmoid**: a Wasserstein critic *scores*, it does
not classify.

```python
def _gradient_penalty(gan, cond, real, fake):
    eps = torch.rand(real.shape[0], 1, device=real.device)
    mixed = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = gan.score(cond, mixed)
    grads = torch.autograd.grad(
        outputs=scores, inputs=mixed,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
```

The penalty pulls the critic's gradient norm toward 1 at points interpolated between
real and fake, enforcing the Lipschitz constraint the Wasserstein distance requires.
`create_graph=True` is essential — we need to differentiate *through* the gradient
computation.

**Five critic steps per generator step** (`n_critic=5`), as WGAN prescribes: the critic
must be near-optimal for its score to approximate the Wasserstein distance.

### The evaluation rule that makes this honest

> **The comparator is not optional.** Any GAN must be judged against cheap augmentation
> — jitter and window warping — on **downstream forecast accuracy**, never on how
> realistic the synthetic series look.

A generator producing convincing series that does not improve forecasting has
contributed nothing. Distributional similarity is precisely the metric that lets a GAN
*appear* successful while being useless. So the cheap methods are implemented first and
the GAN must beat them.

### `augment_training_set(...)`

```python
if kind == "none" or ratio <= 0:
    return x, y, p
...
return torch.cat([x, xs]), torch.cat([y, ys]), torch.cat([p, ps])
```

Synthetic rows are **appended, never substituted**, so the real data is always present.

For the GAN path:

```python
cond  = x.reshape(n_win * n_nodes, cond_dim)
resid = (y - p).reshape(n_win * n_nodes, horizon)
gan = fit_gan(cond, resid, epochs=gan_epochs, seed=seed)
...
ys = ps + fake_resid  # synthetic level = anchor + synthetic correction
```

Flattening (window, node) into independent samples means the generator models a
*district-week*, not the whole graph — so 25 nodes multiply the effective corpus. And
the GAN generates **residuals**, consistent with what the forecasting model predicts.

> Augmentation is applied to the **training split only**. Augmenting validation or test
> would leak synthetic structure into the reported numbers. In `run_single` the
> augmentation call sits after the fold is built and touches only `xtr, ytr, ptr`.

## 6.7 `baselines.py`

Covered in §4.6. The structural point:

```python
from dengue_gnn.experiment import Config, FoldData, _build_fold, _rows_from, fold_split
```

Baselines import the graph models' machinery rather than reimplementing it. Same
windows, same target, same inverse transform, same scoring.

`run_baseline_single` mirrors `run_single` so both write the **same row schema** and
both can be dispatched by the same worker pool. The `base` dict includes fields that
belong to graph models purely so the schemas match:

```python
"shrink_gamma": 1.0,  # no residual shrinkage is applied to baselines
"temporal": "none",
"augment": "none",
```

`_estimator_size` gives tree ensembles a parameter count comparable to a network's:

```python
if hasattr(est, "coef_"):
    return int(np.asarray(est.coef_).size + np.asarray(getattr(est, "intercept_", 0)).size)
if hasattr(est, "estimators_"):
    return int(sum(e.tree_.node_count for e in np.ravel(est.estimators_)))
```

Linear models count coefficients; tree ensembles have no weights, so total node count is
reported instead — the closest honest analogue of capacity, and labelled as such in the
paper rather than passed off as a parameter count.

## 6.8 `provenance.py` — making results reproducible

Rule D2 requires every reported number to have a config, commit SHA, seeds and folds
behind it. In practice that record was being written by hand *after the fact*, which is
how Phase-2 numbers ended up citing scripts that no longer existed (finding F1).

This module writes the record automatically, beside the results CSV, at the time the run
starts.

```python
def git_state(repo=None):
    sha = _run(["git", *cwd, "rev-parse", "HEAD"])
    status = _run(["git", *cwd, "status", "--porcelain"])
    return {
        "commit": sha,
        "branch": _run(["git", *cwd, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }
```

> **`dirty` matters more than the SHA.** A run made against uncommitted changes cannot
> be reproduced from the SHA alone — and that is exactly the situation most of this
> project's results were produced in. Recording it means a reader knows the difference.

`collect_environment()` captures Python version, platform, library versions, thread
counts, CUDA availability. All of these change results without changing code.

Every failure degrades to a recorded `None` rather than an exception:

```python
def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
```

A run that dies because the git query failed is strictly worse than one with an
incomplete manifest.

`write_manifest` is called **twice** in `run_phase3.py` — once before the worker pool
starts (`status: "started"`) and once after (`status: "complete"`, plus row count and
wall time). If the run crashes, the first manifest still stands and says what the run
was and which commit it ran against.

`n_jobs_scheduled` makes truncation detectable: compare it against the distinct
`(label, fold, seed)` triples actually present in the CSV.

## 6.9 `scripts/run_phase3.py` — orchestration

### Parallelism

```python
def _init(npy, adj):
    import torch
    torch.set_num_threads(1)
    global _RAW, _ADJ
    _RAW, _ADJ = load_dataset(npy, adj)[:2]
```

Two things here.

**`torch.set_num_threads(1)`** — counter-intuitive but measured: one thread is **2.7×
faster per epoch** than ten on this workload. The model is ~7k parameters on a
(1, 25, 33) input; thread synchronisation costs more than the arithmetic. So each worker
pins itself to one thread and the process pool fills the cores instead.

**The initializer loads the dataset once per worker process**, into module-level globals,
rather than pickling it with every job. With 128 jobs that is the difference between
loading the array 128 times and 12 times.

The measured effect: 224 jobs in 1,466 s, against Phase 2's 45 jobs in 1,816 s.

### Job construction

```python
for cfg in cfgs_x:
    for fold_idx in range(len(cfg.origins)):
        for seed in cfg.seeds:
            jobs.append((cfg, fold_idx, seed, "model"))
for fold_idx in range(len(cfgs_x[0].origins)):
    jobs.append((cfgs_x[0], fold_idx, 0, "persistence"))
```

A job is a tuple, dispatched by `_job` which switches on the `kind` field. Persistence
runs once per fold — it is deterministic.

### `--labels`, `--note`, `--quick`

- `--labels X_gat_h1,X_gcn_h96` runs only named arms, so one arm can be added to an
  existing study without re-running the rest. Persistence is always kept, since the
  paired tests need it.
- `--note` records what question the run answers, into the manifest.
- `--quick` cuts epochs to 3 and seeds to 1, for smoke tests.

> **Rule D5: never quote `--quick` numbers.** They are intentionally degraded. Not in a
> paper, not in a PR, not in a status report.

## 6.10 `scripts/make_tables.py` and the schema guard

```python
elif list(reader.fieldnames or []) != fields:
    raise SystemExit(f"{path} has a different schema from {paths[0]}; refusing to merge")
```

Merging two CSVs written by different harness versions would average incomparable
quantities. The guard refuses. It fired twice during development, both times correctly.

When `resid_ratio` was added to every row, this guard began refusing to merge
pre-change files with post-change ones. That is the guard **working**, not a
regression.

## 6.11 Testing — 113 tests

The tests do not chase coverage; they encode the properties that must hold.

Representative examples:

**Masking actually masks.** Verify that in the GAT, node 0's output is unaffected by
changing node 2's features when nodes 0 and 2 are not adjacent. If masking were broken,
this would silently pass information along non-edges.

**Peak-week error refuses wrong shapes.** Assert that a 3-D array raises. The bug in
§4.5.1 could not have survived this test.

**A zero λ is a differentiable zero.** Verify `spatial_regularisation(..., lambda_phys=0)`
returns a tensor with a graph, not a Python float.

**Provenance degrades rather than raises.** Monkeypatch the git call to fail; verify a
manifest is still written with `commit: None`.

**An inert constraint does not blow up the adaptive weight.** Verify `adaptive_weight`
returns `current` unchanged when the constraint contributes no gradient.

Two testing lessons learned the hard way:

> **⚠ Trap 1.** A test that constructs a tensor with `torch.rand(...).requires_grad_(True)`
> and then modifies it produces a **non-leaf** tensor, and gradient assertions on it are
> meaningless. Build it as `(torch.rand(...) * 100).requires_grad_(True)` so the
> `requires_grad` is set on the final tensor.

> **⚠ Trap 2.** A patch applied by string replacement can **silently match nothing** if
> a formatter has reflowed the line. If no assertion checks that the replacement
> happened, you get a no-op patch and hours of confusion. Either assert the match, or
> use a tool that fails loudly.

---

# Part VII — Every Experiment, and What It Found

The full log is `docs/EXPERIMENT_LOG.md`, one entry per run with config, commit, seeds,
folds, question, result, and verdict. This part is the narrative.

## 7.1 The arc, in one paragraph

The project started with a plain GCN on a single chronological split; it lost badly to
persistence. Fixing the target parameterisation (residual + log1p) closed most of the
gap. Fixing the protocol (rolling-origin, 8 disjoint folds, seeds recorded separately)
revealed that almost none of the effects were statistically real. A replication at 8
seeds then **overturned the project's headline claim** and simultaneously found its only
statistically significant positive result. A later architecture study found that the
best-looking model was not forecasting at all.

## 7.2 The experiments

### EXP-001 — Baseline v1: single 70/10/20 split
Plain GCN, absolute case scale. GCN test RMSE ≈ 67–72, MAE ≈ 24. Persistence RMSE ≈
59.5, MAE ≈ 13.8. **Persistence wins on every metric.**

The important finding was not the loss but that **validation and test RMSE were
anti-correlated (r = −0.74)**. Selection on validation was worse than random. This is
what forced rolling-origin CV. *Superseded by EXP-002.*

### EXP-002 — Baseline v2: rolling-origin, residual + log1p
3 folds × 3 seeds.

| Model | RMSE | MAE |
|---|---|---|
| Persistence | 44.8 | 15.7 |
| GCN (residual + log) | 45.3 | 15.9 |
| GAT (residual + log) | 45.5 | 15.9 |

**Verdict: the GNN matches the persistence floor; it does not beat it.** The
residual parameterisation does the work (66 → 45); log1p alone *hurts*.

Note persistence = 44.80 here but 55.12 later. Both are correct — the protocol changed
(3 overlapping folds → 8 disjoint folds). Numbers from different protocols are not
comparable, which is exactly why D1 freezes the protocol.

### EXP-003 to EXP-005 — Adaptive graph, λ sweep, combined model
The Phase-2 contributions. These produced the headline claims. All three were later
found to have problems, catalogued as findings F1–F12 in `docs/PHASE2_REVIEW.md`.

### EXP-006 — Protocol widening
8 disjoint origins, plus hyper-parameter axes the earlier study never swept (embedding
width, hidden width). Established the frozen protocol.

### EXP-007 — Non-graph baselines under our protocol
The Comparative Analysis. The key result:

| Model | n | RMSE | vs persistence |
|---|---|---|---|
| **Persistence floor** | 8 | **55.12 ± 41.03** | — |
| Random forest | 24 | 56.31 ± 39.38 | 9/24, p=0.307 |
| XGBoost | 24 | 57.41 ± 40.36 | 7/24, p=0.064 |
| LSTM (no graph) | 24 | 57.95 ± 42.17 | 10/24, p=0.541 |
| Dense GCN, fixed graph | 24 | 64.67 ± 50.99 | 7/24, p=0.064 |
| + adaptive graph | 24 | 60.08 ± 44.25 | 9/24, p=0.307 |
| Ridge regression | 8 | 290.81 ± 675.61 | 2/8, p=0.289 |
| Seasonal naive (52 wk) | 8 | 153.17 ± 79.72 | 0/8, p=0.008 |

Read the ordering carefully. **Random forest beats every graph model.** A tabular
learner with no notion of the graph, no temporal operator, and no deep architecture is
the best non-naive model in the table. That is a finding, and it belongs in the paper.

### EXP-008 — Mechanistic constraints on observables
Growth-band and growth-smoothness penalties, λ chosen from measured loss magnitudes
rather than guesswork (see §6.4). Promising at n=24.

### EXP-009 — The 8-seed replication that overturned the headline

This is the most important experiment in the project. Stage A re-ran four
configurations at **8 seeds (n=64)** rather than 3, because the power analysis said the
effects needed n ≈ 44–54.

| Comparison | Δ RMSE | wins | p |
|---|---|---|---|
| growth constraint vs none | −1.62 | 46/64 | **0.0006** |
| learned graph vs fixed control | −2.78 | 35/64 | 0.532 |
| constraint + curriculum vs none | +2.54 | 32/55 | 0.281 |
| learned graph vs **persistence** | +6.69 | 21/64 | **0.0081** |
| constrained model vs **persistence** | +5.07 | 23/64 | **0.0328** |

Five conclusions:

1. **The adaptive graph does not replicate.** At n=24 it was 17/24, p=0.064 — promising.
   At n=64 it is 35/64, **p=0.53** — indistinguishable from chance. The mean still
   improves by 2.78, but only a minority of runs improve, so the mean is driven by a few
   large gains. **The project's headline contribution is not supported.**
2. **The growth-smoothness constraint is real.** 46/64 wins, **p=0.0006** — the first
   statistically significant positive result in the project.
3. **Persistence significantly beats every model** (p=0.008, p=0.033). Not merely
   unbeaten — the gap is significant.
4. **The curriculum hurts** the constraint that works, reversing the n=24 reading.
5. λ ≈ 0.15 may beat 0.3, but only at n=24, so it needs promotion before it can be
   claimed.

> **Why this experiment was designed the way it was.** The compute budget was split
> deliberately: replication first, breadth second. Spending the same compute on *more
> configurations at n=24* would have produced more p=0.064 results and no way to tell
> which were real. **Two of the four were not.**

### EXP-010 — Temporal encoder, shrinkage, GAN
Contribution (b) and the temporal operator the original design lacked.

### EXP-011 — Reconciliation with prior work
Covered in §4.7.3. The apparent contradiction with Weng et al. was entirely an
aggregation-convention difference: their `evaluation.py` computes mean-of-per-window
RMSE, and persistence computed *their* way scores 38.46 — beating every model in their
published table.

Also contains the residual-parameterisation ablation (§5.2.1).

### EXP-012 — A retracted claim, and how it happened

A 3-fold interim probe suggested window W=26 scored 61.09 and beat persistence
decisively. The full 8-fold × 3-seed run said otherwise:

| Window | wins vs persistence | p | note |
|---|---|---|---|
| W=3 | 12/24 | 1.000 | the best of the three |
| W=26 | 6/24 | 0.023 | **persistence wins significantly** |
| W=39 | — | — | best mean (52.04) but only 5/24 wins, driven by one fold |

The interim claim was **retracted** and logged as such.

> **Lesson LL-022, and it is worth memorising.** *Never report a number from an interim
> probe on a subset of folds.* Probes decide what to run next; only the full protocol
> produces a claim. Three effects in this phase looked significant at small n and
> vanished at full n.

Note also W=39's shape: the best *mean* driven by one fold, with a minority of wins. That
is the signature of an effect that is not real. Always look at the win count, not just
the mean.

### EXP-013 — Published architectures, and the residual collapse
Covered in detail in §5.4 and §5.5. STGAT topped the table while moving 2–5% of the
required distance; GAT lost to the GCN control at matched capacity; nothing beat
persistence significantly.

## 7.3 The twelve review findings

`docs/PHASE2_REVIEW.md` catalogues what went wrong in Phase 2. Every one of these is a
mistake that is easy to make and hard to notice.

| # | Finding | Root cause |
|---|---|---|
| **F1** | Results were not reproducible | Code lived in a git-ignored `scratch/` |
| **F2** | Table 1 mixed Phase-1 and Phase-2 measurements of the same baselines | Two protocols in one table |
| **F3** | Paper numbers did not match the committed CSVs | Manual transcription |
| **F4** | Peak timing error measured nothing | `argmax` on the wrong axis (§4.5.1) |
| **F5** | The regulariser was applied in the wrong space; one term is inert | Magnitude never measured (§6.4) |
| **F6** | The comparison was not controlled | Adjacency *and* normalisation both changed (§3.2) |
| **F7** | No variance recorded; all effects sit inside the noise | Seeds averaged before writing (§4.7.4) |
| **F8** | EXP-005 was not an independent experiment | Reused another run's numbers |
| **F9** | Log comparison columns used different aggregations | No stated convention (§4.7.3) |
| **F10** | `results_logger.py` failed silently into the results | Missing metrics defaulted to `0.0`, gate to `1.0` |
| **F11** | The gate converged to 0.49 in every run | Possibly drift to zero-gradient, not convergence |
| **F12** | Nothing was committed | No branch, no PR, no CI |

**F10 deserves special attention.** The old logger, when a metric column was missing,
wrote `0.0`. And when the learned gate was missing, it wrote `1.0` — a value that reads
as *"the model used pure geography"* rather than *"this was never recorded."*

> A logging failure that writes a plausible wrong number into the file a paper cites is
> **worse than one that crashes**, because nothing downstream can tell the difference.
> The rewritten module either writes the real value or raises.

**F11 is a good example of reading your own logs sceptically.** `σ(g)` landed in
0.4816–0.4990 across every fold and λ, from an initialisation of 0.82. The log
interpreted this as "balanced reliance on geography and learned connectivity." But
landing within 0.02 of exactly 0.5 in nine independent runs is suspicious — it is at
least as consistent with the gate *drifting to a zero-gradient point* as with it
converging to a meaningful value. The response was not to reinterpret it but to record
`gate_sigma` per run so the trajectory becomes visible.

## 7.4 What is actually true about this problem

Stated plainly, because this is what the evidence supports:

1. **Persistence is a strong baseline and significantly beats every model tried.**
   (p = 0.008 for the learned-graph model, p = 0.033 for the constrained model.)
2. **Random forest is the best non-naive model**, beating every graph architecture.
3. **The adaptive graph does not replicate** at adequate sample size.
4. **The growth-smoothness mechanistic constraint is the one real positive result**
   (46/64, p = 0.0006).
5. **Graph attention loses to uniform propagation** when parameter count is held fixed.
6. **The residual parameterisation does nearly all the work** that separates a useless
   model from a floor-matching one.
7. **Fold-to-fold variance dominates every effect.** SD ≈ 39–44 against means ≈ 55–60,
   with effects of 0.5–5. Nothing will be resolvable here without either many more
   folds or a much larger effect.

That is an honest, publishable set of findings — a rigorous negative benchmark plus one
significant positive. It is not the result the proposal hoped for, and reporting it
accurately is the point.

> **Rule D8: negative results ship.** Report a failed contribution with the ablation
> intact. Quietly dropping a row is the only thing here that actually costs marks — and
> in a real venue, it is the thing that gets papers retracted.

---

# Part VIII — Rebuilding From Scratch

Here is the order to build it in. Each step is testable before you move on, which is the
whole point — if you build the model first you will have no way to know whether it works.

## 8.1 Environment

```
python >= 3.11
numpy
torch          (CPU is enough; the model is ~7k parameters)
scikit-learn   (ridge, random forest)
xgboost
pytest
ruff           (linting and formatting)
```

That is the entire dependency list. Notably **not** required: PyTorch Geometric,
`torch_geometric_temporal`, or any graph library. Dense adjacency on 25 nodes is a 25×25
matrix; a graph library buys you nothing and costs you a normalisation you did not
choose (§3.2).

## 8.2 Build order

**Step 1 — Data loading.** Write `load_dataset`. Assert the shape is (459, 25, 11), that
the adjacency has 25 keys, and that node ordering is explicit. *Test:* the adjacency is
25×25, rows sum to 1, and the diagonal is non-zero.

**Step 2 — Metrics.** Write `score` and `peak_week_error` before any model. *Test:*
`score(x, x)` gives RMSE 0; SMAPE is finite when both inputs are zero; `peak_week_error`
raises on a 3-D array; a series shifted by 2 weeks gives a peak error of 2.

**Step 3 — Persistence.** Implement the naive baseline and score it. **You should get
RMSE 55.12** under the 8-fold protocol. If you do not, something upstream is wrong and
you must fix it now — everything downstream is measured against this.

**Step 4 — Fold splitting.** Write `fold_split`. *Test:* the eight test ranges are
exactly 184–217, 218–251, …, 422–455; no train index exceeds any test index; validation
is the last 30 weeks of the training pool.

**Step 5 — Windowing.** Write `_build_fold`. *Test:* shapes are `x (n, 25, 33)`,
`y (n, 25, 3)`, `p (n, 25, 3)`; normalisation statistics come only from training; the
feature-major layout round-trips (reshape to `(n_feat, window)` and back recovers the
original).

**Step 6 — A trivial model.** A single `nn.Linear` from 33 to 3, no graph. Train it.
This validates the whole training and evaluation loop before any graph complexity. If it
does not roughly match persistence, your loop is broken, not your architecture.

**Step 7 — The GCN.** Add `A @ X @ W` with the fixed adjacency. *Test:* changing a
non-neighbour's features does not change a node's output after one layer.

**Step 8 — The adaptive adjacency and gate.** *Test:* `use_adaptive=False` creates zero
extra parameters; the blended adjacency's rows sum to 1; `gate_value()` returns exactly
1.0 in the control.

**Step 9 — The harness.** `run_single`, `rolling_origin`, row emission. *Test:* rows are
one per (fold, seed, horizon) plus a pooled row; nothing is averaged before writing.

**Step 10 — Baselines.** Ridge, random forest, XGBoost, LSTM, seasonal naive — all
sharing the harness.

**Step 11 — Statistics.** `sign_test`, aggregation, table generation.

**Step 12 — Only now**, temporal operators, attention, regularisation, augmentation.

> The ordering is the lesson. **Protocol and metrics before models.** If you build the
> model first, every number you produce is unfalsifiable until the protocol exists, and
> you will have to re-run everything anyway.

## 8.3 The rules, distilled

These are the project's operating rules (`.antigravity/project/rules.md`), and they
generalise to any empirical ML project.

| Rule | Statement |
|---|---|
| **D1** | The protocol is frozen. Changing origins, horizon, or normalisation invalidates every cross-row comparison. |
| **D2** | No number without provenance: a log entry (config, SHA, seeds, folds), a results row, and a **stated aggregation**. |
| **D3** | Code that produces a number is committed *before* the number is quoted. |
| **D4** | Rows are per (fold, seed, horizon). Never aggregate before writing. |
| **D5** | Never quote `--quick` numbers. They are intentionally degraded. |
| **D6** | One owner per notebook at a time. `.ipynb` merge conflicts are not hand-resolvable. |
| **D7** | One variable at a time, each against a control sharing the rest of the pipeline. |
| **D8** | Negative results ship, with the ablation intact. |
| **D9** | Covariates are already lag-shifted. Never apply a second lag. |
| **D10** | `main` is protected. Branch, review, squash-merge. |
| **D11** | The page budget is measured, not estimated. |
| **D13** | Never delegate the novelty. A subtly wrong ODE that trains fine is worse than one that crashes. |
| **D14** | Ship a contribution that measurably improves something under the frozen protocol and a paired test. |

## 8.4 The complete trap list

Every one of these actually happened, or was actively guarded against.

**Data and shapes**
1. Applying a second lag to already-shifted covariates.
2. Node order in the adjacency not matching axis 1 of the array — silently scrambles the graph.
3. Reshaping the flat 33-vector as `(window, n_feat)` instead of `(n_feat, window)`.
4. NaNs in satellite data poisoning every gradient.
5. `keepdim=False` in row normalisation, transposing the division.

**Protocol**
6. Shuffling a time series.
7. Using a single chronological split for model selection (r = −0.74 here).
8. Overlapping test windows inflating apparent sample size.
9. Computing normalisation statistics over the whole array.
10. Looking at the test set more than once.
11. Running deterministic baselines at 3 seeds, inflating *n* with duplicates.

**Training**
12. Forgetting `opt.zero_grad()`.
13. Forgetting `model.eval()` before evaluating.
14. Storing `best_state` without `.clone()` — restores the final weights, not the best.
15. Symmetric padding in a causal convolution — leakage inside the architecture.
16. Choosing a shrinkage or hyper-parameter on the test fold.

**Objective**
17. Applying a non-negativity penalty to residuals, biasing against predicting declines.
18. Choosing λ without measuring the loss terms' magnitudes (17,700× off here).
19. Imposing a constraint at full strength from step one (an optimisation failure, not a constraint failure).
20. Claiming an inert term as a contribution.

**Evaluation**
21. `argmax` on an axis that is not time.
22. Reporting a mean without a win count — a mean driven by one fold is not an effect.
23. Comparing architectures at unmatched parameter counts.
24. **Reporting RMSE for a residual model without checking it moved off the baseline.**
25. Not stating your aggregation convention (1.53× here).
26. Quoting an interim probe on a subset of folds.
27. Comparing numbers produced under two different protocols.

**Engineering**
28. Code that produces a number living in a git-ignored directory.
29. A logger that defaults a missing metric to a plausible number.
30. Merging CSVs with different schemas.
31. Maintaining separate sequential and parallel runners that drift apart.
32. A string-replacement patch that silently matches nothing.
33. Reporting results from a dirty working tree without saying so.

## 8.5 Glossary

**Adjacency matrix** — *N×N* table of which nodes connect to which.
**Ablation** — removing one component to measure its contribution.
**Autocorrelation** — correlation of a series with a time-shifted copy of itself.
**Backpropagation** — the chain-rule algorithm computing gradients through a network.
**Batch** — a group of samples processed together. Here, batch size 1.
**Buffer** — model state that moves with the model but is not trained.
**Causal convolution** — a convolution that cannot see the future; enforced by left-padding.
**Confound** — a second variable that changed at the same time, making attribution impossible.
**Dropout** — randomly zeroing activations during training, as a regulariser.
**Early stopping** — halting when validation stops improving; restore the best weights.
**Epoch** — one pass over the training data.
**Expanding window** — CV where the training set grows with each fold.
**Fold** — one train/test split in cross-validation.
**GAN** — generator and critic trained adversarially.
**Gate** — a sigmoid-valued scalar or vector controlling how much of a signal passes.
**Generation interval** — time from one infection to those it causes; ~3 weeks for dengue.
**Gradient clipping** — rescaling gradients whose norm exceeds a threshold.
**Horizon (H)** — how many steps ahead you forecast. Here 3.
**Leakage** — information from the test period reaching training. The cardinal sin.
**log1p / expm1** — `log(1+x)` and `e^x − 1`; a mutually inverse pair safe at zero.
**Mode collapse** — a GAN producing one output regardless of input.
**Node / edge** — a graph's vertices and connections. Here districts and borders.
**Over-smoothing** — too many graph layers making all node representations identical.
**Paired test** — comparing two methods on the *same* folds and seeds.
**Parameter** — a number learned by gradient descent.
**Persistence** — the naive forecast "next week equals this week."
**Power analysis** — computing the sample size needed to detect an effect.
**Receptive field** — how far information travels; 2 graph layers = 2 hops.
**Regularisation** — anything that trades training fit for generalisation.
**Residual (target)** — predicting the *change* from a baseline rather than the level.
**Rolling origin** — CV over many chronological forecast origins.
**Seed** — the RNG initialiser; different seeds give different results.
**Sign test** — count wins, test against a binomial null.
**SMAPE** — symmetric MAPE, bounded at 200%, finite at zero.
**Softmax** — normalises a vector to positive values summing to 1.
**Window (W)** — how many past weeks the model sees. Here 3.
**z-normalisation** — subtract the mean, divide by the SD.

## 8.6 References

The works this project builds on directly.

**Architectures**
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR. The GCN.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR. GAT, masked attention.
- Wu et al. (2019), *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI. Adaptive adjacency and the gated TCN.
- Bai et al. (2021), *A3T-GCN: Attention Temporal GCN for Traffic Forecasting*. Temporal attention.
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*. The LSTM.

**Training physics-informed models**
- Krishnapriyan et al. (2021), *Characterizing Possible Failure Modes in Physics-Informed Neural Networks*, NeurIPS. Curriculum regularisation.
- Wang, Teng & Perdikaris (2021), *Understanding and Mitigating Gradient Flow Pathologies in PINNs*, SIAM J. Sci. Comput. Gradient-norm loss balancing.

**Generative models**
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*. WGAN-GP.
- Esteban, Hyland & Rätsch (2017), *Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs*. The RCGAN conditioning pattern.

**Domain**
- Weng et al. (2024), *Graph Representation Learning for Dengue Forecasting*, IEEE BigData. The benchmark reproduced here; source of the dataset assembly and lag-shifting.
- Clarke et al. (2024), *OpenDengue*, Scientific Data 11:296. Reserved generalisation set.

**Method**
- Bergmeir & Benítez (2012), *On the use of cross-validation for time series predictor evaluation*, Information Sciences. Why rolling origin.

## 8.7 Where to look in this repository

| Question | File |
|---|---|
| What decisions were made and why | `docs/decisions/` (ADRs) |
| What went wrong in Phase 2 | `docs/PHASE2_REVIEW.md` (F1–F12) |
| Every run, with config and verdict | `docs/EXPERIMENT_LOG.md` |
| How prior work reconciles with ours | `docs/RECONCILIATION_WITH_PRIOR_WORK.md` |
| The dataset in detail | `docs/DATA.md` |
| Which CSV came from which commit | `results/RUNS.md` |
| The operating rules | `.antigravity/project/rules.md` |
| Accumulated lessons | `.antigravity/memory/lessons/` |

## 8.8 The three things to take away

If you remember nothing else:

**1. The protocol matters more than the model.** Most of the effort here went into
evaluation, and every significant finding came from evaluation being right — the
anti-correlated split, the vanishing adaptive-graph effect, the aggregation
discrepancy, the residual collapse. None of those are architecture problems. A good
model measured badly is worth nothing; a mediocre model measured well is a contribution.

**2. Always ask what your metric cannot see.** RMSE cannot tell a forecast from a copy
of the baseline. A mean cannot tell a consistent effect from one lucky fold. A pooled
autocorrelation cannot tell temporal structure from between-district differences. For
every number you report, ask what would look identical if you were completely wrong —
then go and measure *that*.

**3. Report what you found, not what you hoped.** The honest result here is a rigorous
negative benchmark plus one significant positive constraint. That is a genuine
contribution: it tells the next person that on this dataset, at this scale, the graph
does not help and persistence is hard to beat. Nobody learns anything from a table with
the failures quietly removed.

---

*Generated as a teaching document for the CS3631 dengue forecasting project. Every
number in it was recomputed from the repository or read from a committed results file.
Where the project's own documentation and the data disagreed, the recomputed value is
given and the discrepancy noted.*
