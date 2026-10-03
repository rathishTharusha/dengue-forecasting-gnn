A physics-informed loss is not automatically a useful prior. With unobserved compartments, it can regularize the trajectory—or force the network to explain data through the wrong latent mechanism.

## Main limitations when compartments are unobserved

**1. Non-identifiability of latent states and parameters.** If only reported incidence or infectious cases are observed, many combinations of $$S,E,I,R$$ and parameters can produce nearly identical outputs. In a dengue SEIR–SEI model, mosquito abundance, biting rate, human-to-mosquito transmission, mosquito-to-human transmission, reporting fraction, and initial exposed/infectious states can compensate for one another. A low residual therefore does not imply that the inferred exposed or mosquito compartments are correct. Compartment-model studies explicitly warn that unmeasured states must be inferred by the model and that practical non-identifiability can prevent reliable predictions [^1]. Structural-identifiability work recommends adding observations from other states, fixing correlated parameters, or reducing the model [^2].

**2. Confounding with the observation process.** Reported cases are usually a filtered, delayed, and undercounted proxy for infections. If a PINN fits $$y_t\approx\rho\,\text{incidence}_t$$ but $$\rho$$ is unknown, transmission and reporting can trade off. A formal partially observed epidemic example shows that transmission, under-reporting, and prior immunity can be structurally unidentifiable from reported cases alone; complementary prevalence or immunity data restore identifiability [^3].

**3. Latent-compartment hallucination.** The network may produce smooth, plausible $$E(t)$$ or mosquito states that are mathematically compatible with the equations but unsupported by observations. This is especially dangerous when the user interprets those states biologically or uses them to estimate interventions. Physics consistency is a constraint, not validation of the latent state.

**4. Model misspecification.** A standard SEIR/SEIR–SEI system may omit serotype structure, asymptomatic infection, immunity waning, spatial importation, interventions, climate-dependent mosquito mortality, reporting delays, or time-varying behavior. The PINN then learns the best trajectory inside the wrong model class. Lowering the equation residual can worsen forecasting if the real process violates the assumed equations.

**5. Weakly informative physics.** Some equations constrain totals or flows but do not constrain the particular output strongly. For example, many latent trajectories may yield the same weekly incidence. The physics term can make the model appear principled without resolving the actual forecasting ambiguity.

**6. Sparse and irregular data.** Sparse data make it difficult to determine whether a rapid change reflects a true epidemic transition, delayed reporting, a change in surveillance, or noise. PINNs are attractive in this setting—one study reports estimating fixed and time-varying parameters with few data points and incremental learning [^4]—but that result does not remove the underlying identifiability problem.

**7. Positivity and conservation are not guaranteed by residual minimization alone.** A small mean-squared ODE residual can coexist with negative compartments, incorrect population totals, or implausible mosquito-to-human ratios unless these are imposed through parameterizations or additional penalties.

**8. Stiff, multiscale dynamics.** Human infection and mosquito infection can operate on different time scales. Exposed-to-infectious transitions, short mosquito life spans, seasonal forcing, and abrupt interventions can make the residual optimization poorly conditioned. PINN optimization studies identify stiff dynamics and competing gradient behavior as central training difficulties [^5].

## When the physics loss can hurt forecasting

Let

$$
\mathcal L=\mathcal L_{\mathrm{forecast}}+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}.
$$

The physics term hurts when minimizing $$\mathcal L_{\mathrm{phys}}$$ moves the model away from the conditional distribution that generates future observations. The common cases are:

### 1. The epidemiological model is wrong for the deployment setting

If the true outbreak contains a time-varying reporting process, importations, serotype replacement, control interventions, or climate-driven vector dynamics that SEIR–SEI omits, a high $$\lambda_{\mathrm{phys}}$$ suppresses real changes as “unphysical.” The model may forecast the assumed epidemic rather than the observed one.

### 2. The loss is over-weighted relative to data

With large $$\lambda_{\mathrm{phys}}$$, the network may produce a smooth, mechanistically consistent trajectory that misses a real peak, intervention shock, or reporting change. With very small $$\lambda_{\mathrm{phys}}$$, the physics term has little effect and only adds optimization cost. The useful value of $$\lambda_{\mathrm{phys}}$$ depends on scaling, noise, observation density, and whether the equations are trusted.

### 3. The data are noisy but the model is treated as exact

Surveillance counts often contain overdispersion, under-reporting, batching, and revisions. A squared residual against an exact deterministic ODE treats every discrepancy as neural-network error. This can bias the trajectory toward an overly smooth curve and erase genuine short-term predictive signals. A count likelihood plus a probabilistic or relaxed physics term is safer than unweighted MSE when counts are noisy.

### 4. The forecast target is not the state constrained by the equations

A model may forecast weekly reported cases, while the physics constrains latent prevalence or instantaneous incidence. If the observation map, reporting delay, and weekly aggregation are wrong, the network is optimized for a different target. For weekly data, the physics term should usually constrain the seven-day flow or integrated incidence—not equate weekly cases with $$I(t_k)$$.

### 5. Parameters are non-identifiable but jointly optimized

Jointly learning $$\beta$$, $$\gamma$$, mosquito abundance, reporting, and latent initial conditions can yield excellent in-sample fit with unstable parameter combinations. Those combinations may extrapolate very differently. The resulting physics loss can increase confidence without increasing forecast validity. This is the practical consequence of parameter correlations that structural-identifiability analysis is designed to detect [^2].

### 6. The physics is valid only in part of the time or space domain

A fixed transmission rate may be reasonable within a short regime but fail after behavior changes or an intervention. A single global SEIR residual then penalizes the correct regime change. Use time-varying parameters, regime-specific equations, intervention inputs, or a gated physics weight instead of enforcing one immutable law everywhere.

### 7. Gradient conflict and poor conditioning dominate training

The data gradient may ask the network to reproduce a sharp outbreak peak while the physics gradient favors a smoother trajectory. If the gradients point in opposing directions, optimization can stagnate or sacrifice one objective. This is a recognized PINN failure mode in stiff dynamics [^5]. Adaptive weighting, nondimensionalization, compartment-wise scaling, staged training, and residual sampling around peaks can help—but they do not fix a wrong model.

### 8. Spatial coupling is misrepresented

In a spatial dengue model, human mobility, mosquito dispersal, and local host–vector transmission are different mechanisms. If a graph Laplacian is used as a generic diffusion term, it can spread infection too aggressively or in the wrong direction. A graph physics loss should use directed, weighted mobility data where available and preserve local SEIR–SEI flows separately.

## Practical diagnostics before trusting the loss

1. **Run an identifiability analysis** for the actual observed outputs, not the full latent state.
2. **Fit several parameterizations** and check whether equally good fits produce divergent latent states or forecasts.
3. **Compare data-only, physics-only, and hybrid ablations** on strictly chronological holdouts.
4. **Use a realistic observation model**: reporting fraction, delay, weekly aggregation, overdispersion, and missingness.
5. **Inspect compartment plausibility**: positivity, mass balance, mosquito-to-human ratio, and parameter bounds.
6. **Check residuals by regime**: pre-peak, peak, decline, intervention periods, and spatial hotspots.
7. **Add complementary measurements**—serology, mosquito abundance, vector infection, hospitalization, or mobility—when latent states matter. Additional state observations are a principled remedy for non-identifiability [^2][^3].
8. **Tune or anneal the physics weight using future-like validation**, not only training residuals.

The practical rule is: use the physics loss as a calibrated, uncertainty-aware prior rather than an absolute truth. It is most likely to help when the compartment model is reasonably adequate, the observation model is explicit, the learned parameter set is identifiable enough for the forecast target, and the physics/data weights are validated out of sample. It is most likely to hurt when the model is misspecified, latent states are weakly observed, reporting processes are ignored, or the residual is enforced more strongly than the data support.

[^1]: Gallo et al., 2020. Lack of practical identifiability may hamper reliable predictions in COVID-19 epidemic models. Science Advances.

[^2]: Chowell et al., 2022. Structural identifiability analysis of epidemic models based on differential equations: A Primer.

[^3]: Bergström et al., 2025. Identifiability in Epidemic Models with Prior Immunity and Under-Reporting. Bulletin of Mathematical Biology.

[^4]: Xia et al., 2024. Physics-Informed Neural Networks for Infectious Disease Modeling with Limited Data. 2024 IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT).

[^5]: Wang et al., 2021. Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks. SIAM Journal on Scientific Computing.