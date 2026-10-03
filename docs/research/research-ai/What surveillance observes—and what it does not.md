### What surveillance observes—and what it does not

In a standard dengue SEIR–SEI model, the routinely observed quantity is usually **reported new human infection**, not a compartment count. Weekly or monthly case notifications are closest to the flow into human infection, $$
\text{new cases}(t) \approx \rho(t)\,\lambda_h(t)S_h(t),
$$ where $$\rho(t)$$ is a time-varying detection/reporting fraction. They are not a direct census of $$I_h$$, because infections can be asymptomatic, delayed in notification, misclassified, or reported after onset. Forecasting systems commonly assimilate “new infected humans” or reported cases rather than directly observed infectious prevalence. [^1]

| Model quantity | Practical observability |
|---|---|
| $$S_h$$, susceptible humans | **Not directly observed.** It is inferred from population size, prior incidence, seroprevalence, vaccination or immunity assumptions, and model dynamics. |
| $$E_h$$, exposed/incubating humans | **Latent.** Routine notification generally does not identify the infection-to-infectiousness interval for each person. |
| $$I_h$$, infectious humans | **Only indirectly observed.** Confirmed cases, hospitalizations, laboratory positivity, and symptom-onset data provide noisy, delayed proxies; they do not measure all infectious people. |
| $$R_h$$, recovered/immune humans | **Partly measurable but usually incomplete.** Serosurveys can estimate prior infection or immunity, but routine case surveillance does not provide a complete recovered/immune census. |
| $$S_m$$, susceptible mosquitoes | **Partly observed at the population level.** Ovitrap, larval, adult-mosquito, or oviposition indices can inform abundance, but they do not directly enumerate every susceptible mosquito. Mosquito-density surveillance has been used as a model input. [^1]** |
| $$E_m$$, infected but not yet infectious mosquitoes | **Latent in routine surveillance.** It normally requires destructive mosquito testing or intensive entomological sampling and is rarely observed as a continuous time series. |
| $$I_m$$, infectious mosquitoes | **Usually unobserved.** Routine dengue surveillance records human disease, not the prevalence of infectious mosquitoes. The 2026 dengue DINN paper explicitly describes mosquito infection dynamics as rarely observable and reconstructs infected-mosquito trajectories from human incidence. [^2]** |
| $$N_h$$, human population size | **Usually observed or estimated well** from census or administrative data, though the effective population exposed to an outbreak may be smaller than the administrative population. |
| $$N_m$$, mosquito population size | **Estimated from entomological indices**, not directly known continuously; seasonal birth, death, and sampling assumptions matter. |

The most important distinction is between an **observed flow** and a **latent stock**: case counts are observations of new detected infections over an interval, whereas $$S_h,E_h,I_h,R_h,S_m,E_m,I_m$$ are stocks that evolve continuously. Even when a paper labels a variable “infected humans,” the data may represent only reported incidence. One validated dengue forecasting system explicitly modeled clinical reports as a fraction of total new infections and used mosquito-density and meteorological data as additional inputs. [^1]

### How neural and physics-informed methods handle the hidden states

There are four main strategies.

**1. Direct neural forecasting bypasses the compartments.** A conventional ANN, RNN, GRU/LSTM, or graph neural network takes case history plus climate, mobility, spatial, or socioeconomic covariates and predicts future incidence. This can forecast the observed output without estimating $$E_h$$ or $$I_m$$, but it does not make the hidden compartments identifiable or mechanistically interpretable. A model that predicts cases well may still imply implausible mosquito or exposed-human trajectories because no compartment equations constrain it.

**2. State-space/data-assimilation models update hidden states from observed cases.** EnKF/EAKF systems propagate all compartments forward using the mechanistic equations, then use the observed case stream to correct the ensemble. Unobserved states and parameters are updated through cross-covariances between the observed cases and the hidden states. The dengue SEIR–SEI–EnKF study used this architecture and reported better predictions than simpler SEIR- and SIR-EnKF comparators in its tested settings. [^3] A related dengue SIR/EAKF system explicitly assimilated reported human cases while estimating unobserved mosquito and human states and model parameters. [^1]

**3. PINNs represent every compartment as a neural function of time.** The network outputs candidate trajectories such as $$\hat S_h(t),\hat E_h(t),\hat I_h(t),\hat R_h(t),\hat S_m(t),\hat E_m(t),\hat I_m(t)$$. Training minimizes a combined loss:

$$
\mathcal L = \mathcal L_{\text{observations}} + \lambda_{\text{ODE}}\mathcal L_{\text{ODE}} + \lambda_{\text{IC}}\mathcal L_{\text{initial conditions}} + \lambda_{\text{constraints}}\mathcal L_{\text{constraints}}.
$$

The observation term is applied only to available data—typically reported incidence—while the ODE residual term penalizes trajectories that violate the SEIR–SEI equations. Initial conditions, non-negativity, population conservation, and sometimes parameter bounds regularize the solution. General PINN work describes this explicitly: the neural network learns latent ODE solutions while differential-equation residuals and initial/boundary constraints act as physics penalties. [^4]

**4. Hybrid mechanistic-neural models infer hidden states first, then forecast them.** The clearest dengue-specific example I found is the 2026 Dengue-Informed Neural Network. It fits reported incidence from 15 countries with a reduced mechanistic transmission model, reconstructs the unobserved infectious-mosquito trajectory, and then uses that inferred trajectory plus lagged climate variables as input to RNN, GRU, and LSTM models for one-month-ahead vector forecasting. The predicted mosquito dynamics are then fed back into the transmission model for longer-horizon dengue projections. [^2] This is not merely a black-box incidence predictor: the mechanistic model supplies a biological bridge from observed human cases to an inferred vector state. But the inferred $$I_m(t)$$ remains model-dependent; it is not a direct measurement.

### What the physics constraint does—and does not—solve

Physics-informed training improves **regularization**, not observability by itself. It can prevent the network from fitting reported cases with trajectories that violate mass balance or the assumed transition rates, and it can estimate unknown parameters jointly with hidden states. X-TFC, a constrained PINN framework tested on SIR, SEIR, and SEIRS inverse problems, approximated latent solutions, enforced initial conditions analytically, combined data and ODE losses, and recovered parameters from noisy synthetic data. [^4] A related PINN study used infectious data with the SIR equations to estimate time-varying transmission rates and reported short-term forecasting capability. [^5] [^6]

The unresolved issue is **identifiability**. With only one incidence time series, many combinations of $$S_h$$, $$E_h$$, $$I_h$$, mosquito abundance, mosquito infection prevalence, transmission rate, reporting fraction, and initial conditions can generate similar observed cases. A PINN may return one smooth, equation-consistent reconstruction without that reconstruction being uniquely supported by the data. Additional observations—mosquito abundance and infection testing, serology, hospitalization or symptom-onset data, temperature and rainfall, multiple locations, or serotype data—can reduce this ambiguity. The dengue DINN paper makes the same practical point from the other direction: mosquito population size and temperature-dependent traits are difficult to specify and rarely observed, which is precisely why it treats vector dynamics as an inference problem. [^2]

For a dengue SEIR–SEI implementation, the defensible workflow is therefore: treat reported cases as an observation model for new human infections; impose the SEIR–SEI ODEs, initial conditions, positivity, and population constraints; infer hidden states and uncertain parameters jointly; and validate hidden-state predictions against independent entomological or serological data whenever possible. A good incidence forecast alone does not validate the inferred $$E_h$$ or $$I_m$$ trajectories.

[^1]: Chen et al., 2022. An ensemble forecast system for tracking dynamics of dengue outbreaks and its validation in China. PLoS Comput. Biol.

[^2]: Cao et al., 2026. Inferring unobserved vector dynamics for dengue forecasting using physics-informed neural networks and mechanistic transmission models. bioRxiv.

[^3]: Yi et al., 2021. SEIR-SEI-EnKF: a new model for estimating and forecasting dengue outbreak dynamics. IEEE Access.

[^4]: Schiassi et al., 2021. Physics-Informed Neural Networks and Functional Interpolation for Data-Driven Parameters Discovery of Epidemiological Compartmental Models. Mathematics.

[^5]: Millevoi et al., 2023. A Physics-Informed Neural Network approach for compartmental epidemiological models. PLoS Comput. Biol.

[^6]: Grimm et al., 2020. Estimating the time-dependent contact rate of SIR and SEIR models in mathematical epidemiology using physics-informed neural networks. Electronic Transactions on Numerical Analysis.