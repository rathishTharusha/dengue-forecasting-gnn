# Chapter 4 — Machine Learning Foundations

*Prerequisites: Chapter 1. Basic Python. No prior ML assumed.*

This chapter builds supervised learning from nothing, then applies it to time
series — where most of the standard advice is actively wrong. Every concept is
grounded in code from `analysis/lib/adaptive.py`, which you will meet again in
Chapter 8.

---

## 4.1 What a model is

A model is a function with adjustable numbers inside it.

$$\hat{y} = f_\theta(x)$$

- $x$ — the **input** (here: three weeks of case counts for 25 districts).
- $\hat{y}$ — the **prediction** (three weeks ahead, 25 districts).
- $\theta$ — the **parameters**: every adjustable number, collectively.

**Training** means: search for the $\theta$ that makes $\hat{y}$ close to the true
$y$ across many examples. That is the whole of supervised learning. Everything
else is detail about how to search.

The simplest useful $f$ is a linear layer:

$$f_\theta(x) = Wx + b$$

```python
nn.Linear(3, 64)     # takes 3 numbers, returns 64
```

`W` here is 64×3 and `b` is 64 — **256 parameters**, all initialised randomly and
all adjusted by training.

### Why stacking linear layers alone is pointless

If $f_1(x) = W_1 x$ and $f_2(h) = W_2 h$, then

$$f_2(f_1(x)) = W_2 W_1 x = W' x$$

Two linear layers collapse into one linear layer. Depth buys nothing.

The fix is a **non-linearity** between them. The standard choice is ReLU:

$$\text{ReLU}(x) = \max(0, x)$$

```python
h = torch.relu(self.lin_in(x))
```

It is not obvious that something this crude should work. The reason it does:
ReLU makes the network **piecewise linear** — each unit is either "on" or "off",
and different regions of input space activate different subsets of units, so the
network applies a different linear map in each region. With enough units you can
approximate any continuous function this way. That statement is the *universal
approximation theorem*, and it is worth knowing mostly so you know it says nothing
about whether you can *find* those parameters.

ReLU also has a practical virtue: its gradient is exactly 0 or exactly 1, which
neither shrinks nor explodes as it passes back through many layers. Older
alternatives like sigmoid have gradients below 0.25 everywhere, so ten stacked
layers multiply them down to ~10⁻⁶ and learning stops — the *vanishing gradient*
problem.

---

## 4.2 Loss — turning "close" into a number

You cannot optimise "close". You need a single number measuring badness.

**Mean squared error:**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

```python
loss_fn = nn.MSELoss()
```

Squaring does two things. It makes the loss differentiable everywhere (unlike
absolute value, which has a kink at zero), and it **penalises large errors
disproportionately** — an error of 10 costs 100× an error of 1.

That second property is a choice, not a law, and on this dataset it is a
dangerous one. Chapter 3 showed the top 1% of district-weeks carry 18.3% of all
cases. Under MSE those weeks dominate the gradient.

**Huber loss** is the standard response:

$$
\mathcal{L}_\delta(e) =
\begin{cases}
\tfrac12 e^2 & |e| \le \delta \\
\delta(|e| - \tfrac12\delta) & \text{otherwise}
\end{cases}
$$

Quadratic near zero (smooth gradients where it matters), linear far away (an
outlier's gradient is capped). This project offers it as an increment precisely
because of the skew measured in Chapter 3.

**Note what the loss is computed on.** In this project it is not the case count:

```python
pred   = model(fold.x_train[idx], fixed)
target = fold.y_train[idx] - fold.p_train[idx]   # residual over persistence
loss   = loss_fn(pred, target)
```

The network is trained to predict a **correction to last week's value**, in
log space. §4.7 explains why.

---

## 4.3 Gradient descent

We need to find $\theta$ minimising $\mathcal{L}(\theta)$. There is no formula.
There are millions of parameters. So we search.

**The gradient** $\nabla_\theta \mathcal{L}$ is the vector of partial derivatives:
for each parameter, how much the loss changes if that parameter increases
slightly. It points in the direction of steepest *increase*. So step the other
way:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

$\eta$ is the **learning rate**. Too large and you leap past the minimum and
diverge; too small and training takes forever. `lr=1e-3` is the near-universal
default for Adam and is what this project uses.

### Backpropagation

Computing $\nabla_\theta \mathcal{L}$ for every parameter sounds expensive. It is
not, because of the chain rule. If $\mathcal{L}$ depends on $h$ which depends on
$\theta$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial h} \cdot \frac{\partial h}{\partial \theta}$$

Backpropagation applies this layer by layer from the output backwards, reusing
each layer's result for the layer below. The total cost is about the same as one
forward pass.

**PyTorch does this for you.** Every operation on a tensor is recorded in a graph;
`loss.backward()` walks it in reverse and fills in `.grad` on every parameter.
You never write a derivative by hand.

```python
opt.zero_grad()          # 1. clear gradients from the previous step
pred = model(x, fixed)   # 2. forward pass (graph is recorded)
loss = loss_fn(pred, target)
loss.backward()          # 3. backward pass (fills .grad)
opt.step()               # 4. apply the update
```

**`zero_grad()` is not optional.** PyTorch *accumulates* gradients rather than
overwriting them. Omit it and step *n* uses the sum of gradients from steps
1…*n*, which trains something, badly, with no error.

### Mini-batches

Computing the gradient over all training examples is expensive and unnecessary. A
random subset gives a noisy but unbiased estimate:

```python
order = torch.randperm(n)                       # reshuffle each epoch
for start in range(0, n, batch_size):
    idx = order[start : start + batch_size]
    ...
```

One pass over all batches is an **epoch**. The noise is mildly beneficial — it
helps escape poor local minima.

**Reshuffling batch order is fine and does not leak.** Only the *train/test split*
must respect time. Confusing these two is common; §4.6 pins it down.

### Adam

Plain gradient descent uses one learning rate for every parameter. Adam adapts per
parameter, using running averages of the gradient (momentum) and of its square
(scale):

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
```

The practical effect: parameters with consistently small gradients get larger
effective steps. Adam is the default because it works acceptably without tuning.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```

If the gradient vector's norm exceeds 5, rescale it to 5. This bounds a single
step's size, so one pathological batch — say, one containing the week-395
artifact — cannot destroy weights that took hundreds of steps to learn. On
heavy-tailed data this is close to mandatory.

---

## 4.4 Overfitting

Here is the central difficulty of the whole field.

A model with enough parameters can memorise the training data exactly, achieving
zero training loss and learning nothing generalisable. Training loss keeps
falling; performance on unseen data gets worse.

```
        loss
         |  \
         |   \___ training       <- keeps falling
         |    \
         |     \___/‾‾‾ validation  <- turns upward: overfitting starts here
         +--------------------- epochs
```

You cannot detect this from training loss. You need **held-out data** the model
never trains on.

### Three splits, three jobs

| split | used for | seen by the optimiser? |
|---|---|---|
| **train** | computing gradients | yes |
| **validation** | choosing when to stop, and hyperparameters | indirectly |
| **test** | the final reported number | **never** |

The validation set influences the model through your choices, so it is no longer
a clean estimate of generalisation. That is what the third split is for.

**Any number you report from a set you selected on is optimistic.** Chapter 5
shows a published paper where this went wrong in a much more direct way.

### Four defences, all present in this project

**1. Early stopping.**

```python
val = evaluate(model, fold, fixed, "val")["RMSE"]
if val < best_val - 1e-6:
    best_val, waited = val, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
else:
    waited += 1
    if waited >= patience:      # patience = 30
        break

if best_state is not None:
    model.load_state_dict(best_state)
```

Three details are worth study:

- The **snapshot** (`best_state`) — without it you would stop at a *worse* epoch
  than the best one you saw, since you only notice a peak after passing it.
- `.detach().clone()` — without `clone()` you store a reference that later
  training mutates, so your "best weights" silently become the final weights.
- **Patience** — validation RMSE is noisy; stopping at the first uptick stops too
  early. Wait 30 epochs for a new best before giving up.

> The reproduced benchmark's own training loop **keeps the final epoch's
> weights** rather than the best. That is a real difference between their setup
> and ours, and Chapter 9 quantifies it.

**2. Weight decay** (`weight_decay=5e-4`) adds $\lambda\|\theta\|^2$ to the loss,
penalising large weights. Large weights mean sharp functions; sharp functions
memorise. It is a preference for smoothness.

**3. Dropout** (`dropout=0.1`) randomly zeroes 10% of activations during training:

```python
h = torch.relu(a @ self.gc1(h))
h = self.drop(h)
h = torch.relu(a @ self.gc2(h))
```

No unit can rely on any specific other unit being present, so the network cannot
build brittle co-adapted chains. **Dropout must be off at evaluation time** —
that is what `model.eval()` and `model.train()` switch, and forgetting `eval()`
makes your test predictions randomly corrupted.

**4. Small models.** The `STGNN` here has ~10,000 parameters against ~7,000
training windows. On a dataset this size, adding capacity is usually the wrong
move.

---

## 4.5 Turning a time series into training examples

A network needs fixed-size (input, output) pairs. A time series is one long
sequence. **Windowing** converts one into the other.

For window $W=3$ and horizon $H=3$, at each time index $i$:

- input: weeks $[i-3,\, i)$
- target: weeks $[i,\, i+3)$

```
weeks:   ... 10 11 12 | 13 14 15 ...
          └── input ──┘└── target ──┘
```

Slide $i$ forward by one and you get another example. 459 weeks yield ~453
overlapping windows.

```python
ids = list(range(window, n_weeks - horizon))
```

The bounds are exactly right and worth checking yourself: `i` must be ≥ `window`
so the input exists, and < `n_weeks - horizon` so the full target exists.

```python
x = np.stack([scaled[i - window : i].T for i in chosen])
y = np.stack([scaled[i : i + horizon].T for i in chosen])
p = np.stack([np.repeat(scaled[i - 1][:, None], horizon, axis=1) for i in chosen])
```

The `.T` matters. `scaled[i-3:i]` is `(weeks, districts)`; transposing gives
`(districts, weeks)`. Stacking produces `(batch, districts, weeks)` — the node
axis in the middle, which is what graph layers expect.

`p` is the **persistence term**: last week's value (`scaled[i-1]`) repeated `H`
times. Same shape as `y`, so it can be subtracted from it.

### Windows overlap, and that is fine

Consecutive windows share most of their data. This is not leakage — every window's
target is still strictly after its own input. It does mean your effective sample
size is much smaller than your window count, which is why 7,000 windows do not
support a large model.

---

## 4.6 Leakage — the failure mode that produces beautiful wrong results

**Leakage** is any path by which information from the test period reaches the
model. It produces excellent test scores and worthless forecasts.

### Leak 1 — shuffling before splitting

The most common mistake in applied ML:

```python
# CATASTROPHIC on time series
X_train, X_test = train_test_split(X, y, shuffle=True)
```

Shuffling puts week 300 in training and week 299 in test. With lag-1 correlation
of 0.92, the model has essentially been handed the answer. You will measure an
excellent score and have learned nothing.

**On time series, the split must be chronological. Always.**

### Leak 2 — normalising before splitting

Subtler, and it slips past careful people.

```python
# WRONG: mean and std computed over the whole series, including test
z = (x - x.mean()) / x.std()
```

The mean now contains information from the test period. If the test period holds
an outbreak, `x.mean()` is pulled upward, and that shifted value is baked into
every training input. Small, invisible, and it inflates your score.

The correct version:

```python
train_weeks = np.log1p(cases[: train_ids[-1] + 1])       # training weeks ONLY
mean, std = float(train_weeks.mean()), float(train_weeks.std() + 1e-8)
z = (np.log1p(cases) - mean) / std                        # applied to everything
```

Statistics come **only** from training weeks; the transform is then applied to all
weeks including test. That is exactly what deployment looks like: at forecast time
you cannot know the future's mean.

`+ 1e-8` guards against division by zero on a constant series.

**This must be recomputed per fold.** Each fold has a different training period
and therefore different statistics. Computing them once outside the fold loop
leaks the later folds' data into the earlier ones.

### Leak 3 — the closure bug

```python
def pack(chosen, scaled=z):
    # `scaled` is bound at definition time on purpose: closing over the
    # loop's `z` would silently use the last origin's scaling if this
    # were ever called outside the iteration that created it.
```

Python closures capture *variables*, not values. A function defined inside a loop
that refers to `z` sees whatever `z` holds **when called**, not when defined. If
`pack` were ever invoked after the loop, it would silently use the last fold's
normalisation for every fold.

The default-argument trick binds the value at definition time. The comment
explains why — which matters, because the line looks like clutter and someone
would eventually "clean it up".

### Leak 4 — evaluating on training data

The largest one in this project, and it is not ours. The benchmark paper's
"Cross Validated" column is computed on a loader that **includes the training
windows** — roughly 70% of that reported evaluation set is data the model was
fitted on. Chapter 9 documents it in full.

It was not detected by reading the paper. It was found by running the code.

---

## 4.7 Residual learning — the single most effective decision here

Chapter 3 established lag-1 correlation ≈ 0.92 pooled. So a model that predicts
"same as last week" is already good.

Asking a neural network to output the absolute count means asking it to rediscover
that from scratch — burning capacity on relearning something you already know.

**Instead, hand it the persistence prediction and ask only for the correction:**

$$\hat{y}_{t+h} = \underbrace{y_{t-1}}_{\text{persistence}} + \underbrace{f_\theta(x)}_{\text{learned correction}}$$

```python
target = fold.y_train[idx] - fold.p_train[idx]     # train on the residual
...
pred = model(x, fixed) + p                          # add it back at inference
```

**This moved RMSE from ~66 to ~45.** It is by far the largest single improvement
in the project, and it is not an architecture change — it is a change in what the
network is asked to predict.

The general principle is worth keeping: **give the model the easy part for free,
and spend its capacity on what is actually hard.** Residual connections in ResNet
are the same idea at layer scale.

### The interaction with log1p

log1p and residual-over-persistence are used together, and the order matters. The
sequence is:

```
raw counts → log1p → z-score (train stats) → subtract persistence → TRAIN
                                              ↓
                                       add persistence back
                                              ↓
             inverse z-score → expm1 → raw counts → SCORE
```

```python
def inverse(self, values):
    """Undo log1p + z-score, returning raw counts."""
    return np.expm1(np.clip(values * self.std + self.mean, 0, 12))
```

The `clip(0, 12)` is a safety rail: `expm1(12)` ≈ 162,754, already far beyond any
real value, and `expm1` of a large number overflows to infinity, which would
poison every aggregate metric with `nan`.

**A documented finding:** log1p **alone makes things worse.** It only helps in
combination with the residual. Recorded in `docs/decisions/0001`, and a good
reminder that techniques interact — evaluating them one at a time can mislead you
about both.

---

## 4.8 Reproducibility

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

Neural network training is randomised in at least three places: weight
initialisation, batch shuffling, and dropout masks. Fix the seed and the run
repeats exactly.

**One seed is not enough for a result.** Two architectures at one seed each may
differ by more than the same architecture differs from itself across seeds. This
project's protocol uses **3 origins × 3 seeds = 9 runs per configuration**, and
compares distributions rather than points.

There is a related subtlety in the model:

```python
# Learned node embeddings. Only used by 'adaptive' and 'hybrid', but
# always created so parameter initialisation consumes the same RNG draws
# in every arm -- otherwise the arms differ by more than the graph.
self.emb_src = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
```

Parameters are initialised from a shared random stream, in order. Create one fewer
parameter in one arm and **every subsequent parameter in that arm gets different
random values**. Your controlled comparison now differs in the graph *and* in every
initial weight. Allocating the embeddings unconditionally costs a few unused
tensors and preserves the experiment.

This is the kind of detail that separates a comparison you can defend from one you
cannot.

---

## 4.9 Checklist

Before trusting any result on time-series data:

- [ ] Is the train/test split **chronological**?
- [ ] Are normalisation statistics from **training data only**, recomputed **per fold**?
- [ ] Is the reported number from a split used for **no** decisions?
- [ ] Is early stopping restoring the **best** weights, not the last?
- [ ] Is `model.eval()` called before evaluating?
- [ ] Are there **multiple seeds**, with variance reported?
- [ ] Is there a **naive baseline** on the identical splits?

The last one is Chapter 5, and it is where most published work on this dataset
falls down.

---

*Next: [Chapter 5 — Evaluation](05_evaluation.md)*
