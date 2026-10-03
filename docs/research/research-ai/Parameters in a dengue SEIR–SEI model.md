## Parameters in a dengue SEIR–SEI model

A typical model has human states $$S_h,E_h,I_h,R_h$$ and mosquito states $$S_v,E_v,I_v$$. The exact parameter list depends on whether the model includes seasonality, multiple serotypes, aquatic mosquito stages, asymptomatic infection, interventions, or spatial movement.

### Core parameters

| Parameter group | Typical parameters | Role |
|---|---|---|
| Human-to-mosquito transmission | $$a$$, $$b_{hv}$$ | $$a$$ is the mosquito biting rate; $$b_{hv}$$ is the probability that a bite on an infectious human infects the mosquito. |
| Mosquito-to-human transmission | $$b_{vh}$$ | Probability that an infectious mosquito bite infects a susceptible human. |
| Vector abundance | $$N_v/N_h$$ or mosquito recruitment $$\Lambda_v$$ | Scales the force of infection from mosquitoes to humans; sometimes modeled dynamically rather than fixed. |
| Human incubation | $$\sigma_h=1/D_{E_h}$$ | Rate at which exposed humans become infectious; $$D_{E_h}$$ is the intrinsic incubation period. |
| Human recovery | $$\gamma_h=1/D_I$$ | Rate at which infectious humans recover; $$D_I$$ is the infectious-period duration. |
| Human demography | $$\mu_h$$, $$\Lambda_h$$, and sometimes births/deaths by age | Natural human mortality and recruitment; often negligible over a short outbreak but relevant for endemic models. |
| Mosquito incubation | $$\sigma_v=1/D_{E_v}$$ | Rate at which exposed mosquitoes become infectious; $$D_{E_v}$$ is the extrinsic incubation period. |
| Mosquito mortality | $$\mu_v$$ | Per-capita adult mosquito death rate; mosquito life expectancy is approximately $$1/\mu_v$$ under a constant-hazard assumption. |
| Mosquito recruitment | $$\Lambda_v$$, aquatic-stage development, or temperature/rainfall functions | Maintains or drives the vector population. |
| Observation process | Reporting fraction $$\rho$$, delay, overdispersion, sometimes importation | Maps latent infections or incidence to reported cases. |
| Initial conditions | $$S_h(0),E_h(0),I_h(0),R_h(0),S_v(0),E_v(0),I_v(0)$$ | Required to start the dynamical system; often only some are observed. |

A simple force-of-infection specification is

$$
\lambda_h(t)=a(t)b_{vh}(t)\frac{I_v(t)}{N_h(t)},
\qquad
\lambda_v(t)=a(t)b_{hv}(t)\frac{I_h(t)}{N_h(t)}.
$$

The corresponding equations are

$$
\begin{aligned}
\dot S_h&=\Lambda_h-\lambda_hS_h-\mu_hS_h,\\
\dot E_h&=\lambda_hS_h-(\sigma_h+\mu_h)E_h,\\
\dot I_h&=\sigma_hE_h-(\gamma_h+\mu_h)I_h,\\
\dot R_h&=\gamma_hI_h-\mu_hR_h,\\
\dot S_v&=\Lambda_v-\lambda_vS_v-\mu_vS_v,\\
\dot E_v&=\lambda_vS_v-(\sigma_v+\mu_v)E_v,\\
\dot I_v&=\sigma_vE_v-\mu_vI_v.
\end{aligned}
$$

Some dengue models replace $$\Lambda_v$$ and constant $$\mu_v$$ with climate-dependent vector dynamics. A recent Dhaka model explicitly included aquatic and adult mosquito stages, cross-immunity, and mosquito development, behavior, and mortality dependent on temperature, rainfall, and humidity [^1]. That is why parameter lists from different dengue papers are not directly interchangeable.

## How published dengue studies select or estimate them

**Literature or laboratory values.** Incubation periods, infectious duration, mosquito survival, biting behavior, and vector competence are commonly taken from biological or clinical studies, then held fixed or varied within plausible ranges. This is defensible when the parameter is independently measured, but it does not guarantee that the value transfers to a different serotype, vector population, temperature regime, or city. A dengue sensitivity analysis found that mosquito survival, biting behavior, and the duration of human infectiousness strongly influence transmission, while also emphasizing that field knowledge for these processes remains limited [^2].

**Hybrid estimation.** A common practice is to fix biologically supported parameters and estimate the remaining parameters by fitting reported cases. For example, the Cali study first estimated some parameters from mosquito biology and the literature, then estimated the rest by least squares against 2010 reported dengue cases [^3]. This is often the most practical approach for a local model because it reduces the number of weakly identifiable parameters.

**Likelihood or Bayesian calibration.** More complex models can calibrate transmission, reporting, vector, and climate-response parameters jointly against cases, seroprevalence, mosquito counts, or multiple data streams. The Dhaka model calibrated to dengue seroprevalence and compared simulations with epidemiological data [^1]. Bayesian or simulation-based methods are especially useful when uncertainty, reporting undercount, or multiple parameter combinations matter.

**Sensitivity-guided selection.** Sensitivity analysis is often used before calibration to determine which parameters deserve local estimation and which can reasonably remain fixed. This prevents spending computation on parameters that the available data cannot identify and exposes which biological measurements would most reduce uncertainty [^2].

## Fixed parameters versus jointly learned PINN parameters

Both strategies are used. A PINN can hold $$\boldsymbol\phi$$ fixed and learn only the state trajectory $$\hat{\mathbf x}_\theta(t)$$, or it can make $$\boldsymbol\phi$$ trainable and optimize it jointly with the neural-network weights:

$$
\min_{\theta,\boldsymbol\phi}
\;\lambda_d\mathcal L_d(\theta,\boldsymbol\phi)
+\lambda_p\mathcal L_p(\theta,\boldsymbol\phi)
+\lambda_0\mathcal L_0(\theta,\boldsymbol\phi).
$$

For an ODE $$\dot{\mathbf x}=\mathbf f(t,\mathbf x;\boldsymbol\phi)$$, the physics term is

$$
\mathcal L_p
=\frac{1}{N_c}\sum_j
\left\|
\partial_t\hat{\mathbf x}_\theta(t_j)
-\mathbf f\big(t_j,\hat{\mathbf x}_\theta(t_j);\boldsymbol\phi\big)
\right\|^2.
$$

A positive parameterization such as $$\phi_m=\exp(\eta_m)$$ or a bounded sigmoid transform is usually preferable for rates and probabilities.

### Fixed-from-literature parameters

**Advantages**

- Reduces the inverse problem and improves identifiability.
- Uses independent biological knowledge rather than asking case counts to identify every mechanism.
- Makes the PINN easier to train and the fitted trajectory easier to interpret.
- Avoids compensating errors—for example, an unrealistically high biting rate offset by an unrealistically low transmission probability.

**Disadvantages**

- Literature values may not match local vector competence, serotype, climate, mosquito age structure, or intervention conditions.
- Mis-specified fixed parameters create systematic residuals and can bias the parameters that remain trainable.
- Uncertainty is understated if fixed values are treated as exact.
- Time-varying quantities such as contact or biting rates cannot be represented unless they are explicitly parameterized as functions.

### Jointly learned parameters

**Advantages**

- Adapts the model to the observed location, outbreak, reporting system, and time period.
- Can estimate otherwise unobserved quantities such as transmission rates, initial infected populations, reporting fractions, or time-varying contact rates.
- Uses mechanistic equations as regularization, so it is usually less unconstrained than fitting a black-box neural network to cases alone.
- Supports inverse problems with sparse or noisy observations. Epidemiological PINN work has used the approach to learn parameters and disease progression from data, and has extended SIR/SEIR methods to time-dependent contact rates [^4][^5].

**Disadvantages**

- Identifiability can be poor: case counts alone may not distinguish $$a$$ from $$b_{vh}$$, $$b_{hv}$$, mosquito abundance, or the reporting fraction.
- The data and physics losses can pull the optimizer in different directions; parameter estimates can depend strongly on loss weights, initialization, collocation design, and scaling [^6].
- A good trajectory fit does not prove that each learned parameter is biologically correct; several parameter sets can generate nearly identical incidence curves.
- Joint training can be unstable for stiff or multi-timescale vector–host systems and may require parameter bounds, priors, staged training, or multiple independent data streams.
- Without uncertainty quantification, a PINN often returns a point estimate where a posterior distribution is more appropriate.

## Practical recommendation for dengue SEIR–SEI PINNs

A defensible default is **partially constrained joint learning**:

1. Fix or tightly prior-constrain parameters with strong external evidence, such as plausible incubation and infectious-period ranges.
2. Learn location-specific transmission, reporting, initial conditions, and possibly mosquito abundance or vector recruitment.
3. Estimate $$a b_{vh}$$ and $$a b_{hv}$$ as identifiable composite parameters unless mosquito or entomological data can separate them.
4. Use positivity and biologically plausible bounds for all rates and probabilities.
5. Add mosquito counts, seroprevalence, climate, or vector-competence data if the goal is to identify mosquito-specific parameters.
6. Report sensitivity and uncertainty, not only the best-fitting trajectory.

This balances biological grounding with local adaptation. Fixing every parameter is safer computationally but can transfer the wrong ecology; learning every parameter is flexible but often overclaims what case data can identify.

[^1]: Paul et al., 2025. A Climate-Driven Mechanistic Transmission Model to Characterize Dengue Epidemiology in Dhaka, Bangladesh. medRxiv.

[^2]: Ellis et al., 2011. Parameterization and sensitivity analysis of a complex simulation model for mosquito population dynamics, dengue transmission, and their control. American Journal of Tropical Medicine and Hygiene.

[^3]: Arias et al., 2018. Estimación de los parámetros de dos modelos para la dinámica del dengue y su vector en Cali, Colombia. Ingeniería y Ciencia.

[^4]: Shaier et al., 2021. Data-driven approaches for predicting spread of infectious diseases through DINNs: Disease Informed Neural Networks.

[^5]: Grimm et al., 2020. Estimating the time-dependent contact rate of SIR and SEIR models in mathematical epidemiology using physics-informed neural networks. Electronic Transactions on Numerical Analysis.

[^6]: Whitman et al., 2025. Physics-Informed Neural Network Frameworks for the Analysis of Engineering and Biological Dynamical Systems Governed by Ordinary Differential Equations. arXiv.org.