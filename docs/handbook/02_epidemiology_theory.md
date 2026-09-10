# Chapter 2 — Epidemiological Modelling

*Prerequisites: high-school algebra. Calculus notation is explained as it appears.*

This chapter builds compartmental epidemic models from nothing. You need it for
two reasons: the "physics-informed loss" contribution is built on these
equations, and two of the four papers this project reproduces are entirely
mathematical.

---

## 2.1 The idea of a compartmental model

Imagine tracking an epidemic. You could simulate every individual — who they met,
when, whether they were infected. That is agent-based modelling and it needs data
nobody has.

The compartmental approach does something cruder and far more useful. Sort the
population into a handful of **compartments** by disease status, and track only
*how many people are in each*:

- **S** — Susceptible: not infected, can be.
- **E** — Exposed: infected, but not yet infectious (incubating).
- **I** — Infectious: infected and able to transmit.
- **R** — Removed: recovered and immune, or dead.

Everybody is in exactly one compartment, so $S + E + I + R = N$, the total
population. People flow between compartments: S → E → I → R.

The model is a set of rules for how fast people flow along each arrow.

### Rates, and what $dS/dt$ means

The notation $\frac{dS}{dt}$ means "the rate at which $S$ is changing per unit
time". If $\frac{dS}{dt} = -100$, the susceptible population is shrinking by 100
people per day.

You do not need to *solve* calculus to use this. You need to read expressions of
the form

$$\frac{dX}{dt} = (\text{inflows}) - (\text{outflows})$$

as bookkeeping: the change in a compartment is what comes in minus what goes out.
A numerical solver does the rest.

---

## 2.2 The SEIR model

This is the model in Gopalakrishnan's lecture notes, one of the four sources
reproduced in `reproduction/`.

### Deriving the equations

**S → E.** A susceptible person becomes exposed by contacting an infectious one.
Let $\beta$ be the *transmission rate*: the number of contacts per person per unit
time, multiplied by the probability a contact transmits. The chance a given
contact is with an infectious person is $I/N$. So susceptible individuals leave S
at rate

$$\lambda = \frac{\beta I}{N}$$

and the total flow out of S is $\lambda S$. This is the only **non-linear** term
in the model — it multiplies two state variables together — and it is the source
of all the interesting behaviour.

**E → I.** Exposed people become infectious at rate $\sigma$. If the incubation
period averages 5 days, $\sigma = 1/5$ per day. Flow: $\sigma E$.

**I → R.** Infectious people recover at rate $\gamma$. If people are infectious
for 10 days, $\gamma = 1/10$. Flow: $\gamma I$.

Collecting these:

$$
\begin{aligned}
\frac{dS}{dt} &= -\frac{\beta I S}{N} \\
\frac{dE}{dt} &= \frac{\beta I S}{N} - \sigma E \\
\frac{dI}{dt} &= \sigma E - \gamma I \\
\frac{dR}{dt} &= \gamma I
\end{aligned}
$$

Notice the terms cancel in pairs: everything leaving S enters E, everything
leaving E enters I. Adding all four equations gives zero — the population is
conserved. **This is a useful check on any implementation.**

### Working in proportions

Dividing by $N$ and writing $s = S/N$, $e = E/N$, $i = I/N$, $r = R/N$:

$$
\begin{aligned}
\frac{ds}{dt} &= -\beta i s, &
\frac{de}{dt} &= \beta i s - \sigma e, \\
\frac{di}{dt} &= \sigma e - \gamma i, &
\frac{dr}{dt} &= \gamma i
\end{aligned}
$$

Now $s + e + i + r = 1$ and the population size has vanished. This is the form
implemented in `crosscheck/lib/xcheck/seir.py`:

```python
def seir_rhs(t, y, beta, sigma, gamma):
    """Right-hand side of the proportion-form SEIR system."""
    s, e, i, r = y
    return np.array([
        -beta * i * s,                 # ds/dt
        -sigma * e + beta * i * s,     # de/dt
        -gamma * i + sigma * e,        # di/dt
        gamma * i,                     # dr/dt
    ])
```

The function returns the *derivatives*, not the next state. A solver
(`scipy.integrate.solve_ivp`) calls it repeatedly to step forward in time. This
"right-hand side" convention is universal in ODE libraries.

### R₀ — the basic reproduction number

**Definition.** $R_0$ is the average number of secondary infections produced by
one infectious individual introduced into a wholly susceptible population.

The intuition: if $R_0 > 1$ each infection more than replaces itself and the
epidemic grows; if $R_0 < 1$ it dies out.

**Deriving it.** Add the $e$ and $i$ equations:

$$\frac{d(e+i)}{dt} = \beta i s - \gamma i = (\beta s - \gamma)\, i$$

The infected population $e + i$ grows when $\beta s - \gamma > 0$, i.e. when

$$\boxed{R_0 = \frac{\beta s_0}{\gamma} > 1}$$

That is the whole derivation. Note it also tells you *why*: $1/\gamma$ is how long
you stay infectious, $\beta$ is how fast you transmit, so $\beta/\gamma$ is how
many people you infect — scaled by the fraction $s_0$ who are still susceptible.

**Vaccination follows immediately.** Vaccinating a fraction $v$ moves them from S
to R, so $s_0 \to s_0(1-v)$ and $R_0 \to R_0(1-v)$. Setting that below 1:

$$v^* = 1 - \frac{\gamma}{\beta s_0}$$

is the **herd immunity threshold**.

```python
def seir_r0(beta, gamma, s0=1.0, vaccinated=0.0):
    """R0 = beta * s0 * (1 - v) / gamma."""
    return beta * s0 * (1.0 - vaccinated) / gamma
```

### Equilibria

An **equilibrium** is a state where nothing changes: every derivative is zero.
Setting all four right-hand sides to zero forces $e = i = 0$ — the only
equilibria of basic SEIR are **disease-free**. The epidemic always burns out.

Add a trickle of infected travellers, though, and it does not:

```python
def seir_rhs_influx(t, y, beta, sigma, gamma, a, b):
    """SEIR with influx `a` into S and `b` into E."""
    s, e, i, r = y
    return np.array([
        -beta * i * s + a,
        -sigma * e + beta * i * s + b,
        -gamma * i + sigma * e,
        gamma * i - (a + b),
    ])
```

With $a = 0.005$, $b = 0.001$ the infection settles at roughly 5% of the
population and never vanishes — an **endemic equilibrium**. This is closer to
dengue's actual behaviour in Sri Lanka.

---

## 2.3 SEIR–SEI: adding the mosquito

Dengue does not transmit human-to-human. A model without mosquitoes cannot
represent it. Phaijoo & Gurung's SEIR–SEI model — the second mathematical paper
reproduced here — adds a vector population.

**Hosts** (humans) get four compartments: $S_h, E_h, I_h, R_h$.
**Vectors** (mosquitoes) get three: $S_v, E_v, I_v$.

Why no $R_v$? A mosquito never recovers. Its infectious period ends only when it
dies — mosquito lifespan (~2 weeks) is short relative to the infection.

### The equations

In proportion form, with $s_v = 1 - e_v - i_v$:

$$
\begin{aligned}
\frac{ds_h}{dt} &= \mu_h(1 - s_h) - \alpha s_h i_v \\
\frac{de_h}{dt} &= \alpha s_h i_v - \beta e_h \\
\frac{di_h}{dt} &= \nu_h e_h - \gamma i_h \\
\frac{de_v}{dt} &= \delta s_v i_h - (\epsilon + \nu_v) e_v \\
\frac{di_v}{dt} &= \nu_v e_v - \epsilon i_v
\end{aligned}
$$

**Read the cross-coupling.** Hosts are infected by $i_v$ (infectious mosquitoes),
not by other hosts. Mosquitoes are infected by $i_h$. The two populations only
influence each other through those two terms — that is the mosquito-mediated
transmission cycle from Chapter 1, written down.

### The composite parameters

The paper compresses nine biological parameters into five symbols:

| symbol | definition | meaning |
|---|---|---|
| $\alpha$ | $b\,\beta_h\,\pi_v / (N_h \mu_v)$ | vector→host transmission force |
| $\beta$ | $\nu_h + \mu_h$ | rate of leaving host-exposed |
| $\gamma$ | $\gamma_h + \mu_h$ | rate of leaving host-infectious |
| $\delta$ | $b\,\beta_v$ | host→vector transmission force |
| $\epsilon$ | $\mu_v$ | vector death rate |

where $b$ is the biting rate, $\beta_h$/$\beta_v$ are transmission probabilities
in each direction, $\mu_h$/$\mu_v$ are death rates, $\nu_h$/$\nu_v$ are incubation
rates, $\gamma_h$ is host recovery, $\pi_v$ is mosquito recruitment.

Note $\beta$ and $\gamma$ are "rate of leaving the compartment by *any* route",
which is why background mortality $\mu_h$ appears in both.

### R₀ via the next-generation matrix

The simple $\beta/\gamma$ argument does not work here, because infection takes two
steps. We need the general machinery.

**The method.** Linearise the infected subsystem near the disease-free
equilibrium and split it into two matrices:

- $\mathbf{F}$ — **new infections** (transmission).
- $\mathbf{V}$ — **transitions** (progression, recovery, death).

Then $R_0 = \rho(\mathbf{F}\mathbf{V}^{-1})$, the **spectral radius** (largest
absolute eigenvalue) of the next-generation matrix.

The interpretation is worth having: $\mathbf{V}^{-1}$ encodes expected time spent
in each infected state; $\mathbf{F}$ converts that into new infections. Their
product maps one generation of infections to the next, and its dominant
eigenvalue is the per-generation multiplication factor.

With state order $(e_h, e_v, i_h, i_v)$:

$$
\mathbf{F} = \begin{pmatrix} 0&0&0&\alpha \\ 0&0&\delta&0 \\ 0&0&0&0 \\ 0&0&0&0 \end{pmatrix},
\qquad
\mathbf{V} = \begin{pmatrix} \beta&0&0&0 \\ 0&\epsilon+\nu_v&0&0 \\ -\nu_h&0&\gamma&0 \\ 0&-\nu_v&0&\epsilon \end{pmatrix}
$$

Only two entries of $\mathbf{F}$ are non-zero — the two transmission directions.

Working the algebra gives the closed form:

$$\boxed{R_0 = \sqrt{\frac{\alpha\,\delta\,\nu_h\,\nu_v}{\beta\,\gamma\,\epsilon\,(\epsilon + \nu_v)}}}$$

**Why the square root?** Because one "generation" in the matrix sense is a single
host→vector *or* vector→host step, but an epidemiological generation is a full
round trip. $R_0$ is the geometric mean of the two half-steps.

Both routes are implemented, and a test asserts they agree:

```python
def seir_sei_r0(p, method="closed_form"):
    if method == "closed_form":
        num = p.alpha * p.delta * p.nu_h * p.nu_v
        den = p.beta * p.gamma * p.epsilon * (p.epsilon + p.nu_v)
        return float(np.sqrt(num / den))
    if method == "spectral":
        f, v = seir_sei_next_generation(p)
        return float(max(abs(np.linalg.eigvals(f @ np.linalg.inv(v)))))
```

They agree to 1.1 × 10⁻¹⁶ — floating-point identical. Implementing a formula two
independent ways and checking agreement is a technique worth internalising.

---

## 2.4 Sensitivity analysis — which parameters actually matter

Suppose you can spend money on exactly one intervention. Should you reduce biting
(bed nets), kill adult mosquitoes (spraying), or reduce breeding sites?

**Normalised forward sensitivity index:**

$$\Upsilon^{R_0}_{q} = \frac{\partial R_0}{\partial q} \cdot \frac{q}{R_0}$$

This is a **percentage-to-percentage** measure: an index of $+0.5$ means a 1%
increase in $q$ raises $R_0$ by 0.5%. Normalising by $q/R_0$ makes indices
comparable across parameters with different units.

### Deriving them by hand

Because $R_0$ is a square root, every index is half the log-derivative of $R_0^2$,
which makes them all elementary:

- $\pi_v$, $\beta_h$, $\beta_v$ enter $R_0^2$ linearly → index $+\tfrac12$ each.
- $b$ enters **twice** (through both $\alpha$ and $\delta$) → index $+1$.
- $\mu_v$ enters as $1/(\mu_v^2(\mu_v+\nu_v))$ → index $-\tfrac12\!\left(2 + \tfrac{\mu_v}{\mu_v+\nu_v}\right)$.

Evaluated at the paper's parameters:

| parameter | index | reading |
|---|---|---|
| $b$ (biting rate) | **+1.000** | most positive — 1% more biting → 1% higher R₀ |
| $\mu_v$ (mosquito death) | **−1.318** | most negative — killing mosquitoes is the strongest lever |
| $\nu_v$ | +0.318 | |
| $\pi_v, \beta_h, \beta_v$ | +0.500 | |
| $\gamma_h$ | −0.500 | |
| $\nu_h$ | +0.000138 | negligible |
| $\mu_h$ | −0.000208 | negligible |

**The practical conclusion:** target the mosquito. Biting rate and mosquito
lifespan dominate; human-side parameters are three orders of magnitude weaker.

**The modelling conclusion for this project:** a physics-informed loss that
spends capacity constraining $\nu_h$ is spending it where $R_0$ cannot feel it.

### Three defects we found in the source paper

All nine indices reproduce to six decimals — but only from the paper's *Section 4
simulation* values, not from the "Baseline Values" column printed beside them in
Table 1. Recomputing from that column gives $\mu_v = -1.085$ against the printed
$-1.318$.

Four parameters are given two different values by the same paper. For $\pi_v$ and
$\beta_v$ it makes no difference (their indices are $+\tfrac12$ for *any* positive
value, which is likely why it survived review). For $\mu_v$ and $\nu_h$ it does.

Two further defects:

- The Section-4 parameters give **R₀ ≈ 0.78 < 1** — a disease-free regime by the
  paper's own theorems — yet its Figures 2–3 show an outbreak.
- The printed endemic equilibrium is **not a fixed point** in its vector
  components. As printed $e_v^*/i_v^* = \epsilon$, but $di_v/dt = 0$ requires
  $i_v = (\nu_v/\epsilon)e_v$, i.e. $\epsilon/\nu_v$. These agree only if
  $\nu_v = 1$, and $\nu_v = 0.1428$. Solving the vector block directly gives

  $$e_v = \frac{\delta\, i_h\, \epsilon}{(\epsilon + \nu_v)(\epsilon + \delta\, i_h)}, \qquad i_v = \frac{\nu_v}{\epsilon} e_v$$

  which has zero residual and matches an independent numerical root-find.

None of the three invalidates the paper's conclusions. All three are recorded in
`crosscheck/FINDINGS.md` as F4.2–F4.4.

---

## 2.5 A subtlety worth understanding: reparameterisation

This project's own `src/dengue_gnn/seir.py` computes $\mu_v$'s sensitivity as
**−0.818**, not the paper's **−1.318**. The difference is exactly 0.5, and
**neither is wrong**.

The paper parameterises with a fixed *recruitment rate* $\pi_v$, so
$\alpha = b\beta_h\pi_v/(N_h\mu_v)$ — meaning $\mu_v$ appears in $\alpha$ too, and
picks up an extra $-\tfrac12$.

`src/` parameterises with a fixed *standing ratio* $m = N_v/N_h$, so
$\alpha = b\beta_h m$ is independent of $\mu_v$.

They answer different questions:

- **Paper:** "if mosquitoes die faster while births continue at the same rate,
  how does R₀ change?" — the standing population *also* shrinks, so the effect
  looks twice as strong.
- **`src/`:** "if mosquito lifespan shortens but the population size is held
  fixed, how does R₀ change?" — isolates lifespan alone.

The second is what a growth-rate bound needs, which is why the code uses it. The
lesson generalises: **a sensitivity index is only meaningful once you say what is
held constant.** Two correct analyses can disagree by a factor of two purely
through parameterisation.

---

## 2.6 From ODEs to a neural-network loss

The proposed contribution (a) was a "physics-informed" loss. The idea: penalise
the network when its forecasts imply epidemiologically impossible dynamics.

**The obstacle.** The SEIR–SEI model has five state variables. Our data has
**one** — reported cases, a noisy proxy for $i_h$. We never observe $e_h$, $s_v$,
$e_v$ or $i_v$. A residual on the full ODE system cannot be evaluated.

**The workaround the project used:** constrain an *observable consequence*
instead. The model implies a maximum plausible exponential growth rate, so
penalise forecasts implying faster growth than the epidemiology permits:

$$\lambda = \text{spectral abscissa of } (\mathbf{F} - \mathbf{V})$$

which is the largest real part of the eigenvalues of $\mathbf{F}-\mathbf{V}$, and
gives the initial exponential growth rate near the disease-free equilibrium. The
weekly log-growth ceiling is $7\lambda$.

```python
def growth_rate(p):
    """Initial exponential growth rate per day: spectral abscissa of F - V."""
    F, V = next_generation_matrices(p)
    return float(np.max(np.linalg.eigvals(F - V).real))
```

**And then the honest part.** Experiment EXP-014 found the constant used for this
ceiling was **wrong by 3.4×**. It had used $\ln(R_0)/GI$, which assumes every
onward infection happens at exactly the mean generation interval. With the
correctly derived ceiling, the constraint band became so wide that **the term
became inert** — its entire apparent benefit had come from the wrong constant.

That is the current state of contribution (a): the mechanism is understood, the
correct bound is derived, and it does not constrain anything at realistic $R_0$.

---

## 2.7 What to take forward

1. Compartmental models are **bookkeeping**: inflow minus outflow.
2. $R_0 = \rho(\mathbf{F}\mathbf{V}^{-1})$ generalises the simple $\beta/\gamma$
   argument to multi-stage, multi-species systems.
3. Dengue needs a **vector compartment**; a host-only SEIR cannot represent it.
4. **Biting rate and mosquito mortality dominate** R₀; human-side parameters are
   negligible.
5. **Unobserved compartments are the binding constraint** on physics-informed
   losses here.
6. Implement every formula twice, independently, and assert agreement.
7. A sensitivity index is meaningless without stating what is held fixed.

---

*Next: [Chapter 3 — The Dataset](03_the_dataset.md)*
