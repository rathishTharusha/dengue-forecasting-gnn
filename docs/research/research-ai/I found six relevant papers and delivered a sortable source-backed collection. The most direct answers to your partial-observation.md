I found six relevant papers and delivered a sortable source-backed collection. The most direct answers to your partial-observation question are **EINNs**, the **Kharazmi et al. PINN framework**, and **algebraically observable PINNs**. EINNs explicitly assumes that many epidemic states are latent: its SEIRM time-module network outputs all states, fits the observed mortality, and minimizes the ODE residual; it then transfers the learned latent dynamics to an RNN that can ingest exogenous signals. [^1] [^1] Kharazmi et al. use the same core idea—network outputs for observed and unobserved compartments, an observation-mismatch loss plus ODE residual loss, and neural representations for time-varying parameters—but show that sparse observations can leave hidden trajectories and parameters substantially non-identifiable. [^2] [^2] [^2]

The generic physics-informed objective is:

$$
\mathcal L = \lambda_{data}\,\mathcal L_{data}(H(y_{NN}),y_{obs}) + \lambda_{ODE}\,\frac{1}{n_c}\sum_j\left\|\frac{d\hat y_{NN}}{dt}(t_j)-f(\hat y_{NN}(t_j),\theta(t_j))\right\|^2 + \lambda_{IC}\,\mathcal L_{IC},
$$

where $$\hat y_{NN}$$ contains the full compartment trajectory, $$H$$ selects the observed compartments or maps latent states to reported incidence, and $$f$$ is the epidemiological ODE right-hand side. The network is penalized only for mismatch on available observations, but the ODE residual is evaluated for every compartment. In the SEIR example of Komatsu, only $$I(t)$$ is observed; the loss nevertheless produces $$S,E,I,R$$, with the equation residual enforcing all four SEIR equations. [^3] [^3]

The papers use several refinements:

- **Joint latent-state and parameter inference.** The network learns hidden compartments and unknown or time-varying transmission parameters together. This is the main strategy in Kharazmi et al. and the EINN time module. [^2] [^1]
- **Algebraic observability.** Komatsu derives algebraic relations from the SEIR equations—for example, $$\dot I=\epsilon E-\gamma I$$ allows $$E$$ to be expressed using observed $$I$$ and its derivative—and uses those relations to create pseudo-observations for otherwise hidden states before training the PINN. [^3]
- **Additional mechanistic constraints.** EINNs adds monotonicity penalties, such as penalizing $$dS/dt>0$$ and $$dR/dt<0$$ when the model assumes susceptible depletion and recovery accumulation. [^1]
- **Constrained neural expressions.** X-TFC/PINN methods build initial conditions analytically into the neural representation, then minimize both data and ODE-residual losses while estimating parameters. [^4] [^4]
- **Hybrid transfer to forecasting networks.** EINNs first learns latent dynamics with a time-only PINN, then transfers ODE-gradient representations to an RNN that can use mobility, symptoms, hospitalization, or other data absent from the ODE. [^1] [^1]

The scientific limitation is identifiability, not the ability to fit the equations. With only one observed incidence curve, multiple combinations of susceptible population, exposed population, infectious population, transmission rate, reporting fraction, and initial conditions can produce similar outputs. Kharazmi et al. explicitly report that limited observations can yield large uncertainty in inferred hidden dynamics and time-varying parameters. [^2] [^2] Komatsu’s algebraic-observability approach improves the inverse problem only when the necessary derivatives, parameters, conservation laws, and model structure are sufficiently known; its proof of concept used synthetic SEIR data, and the authors note that derivative values were supplied in the setup. [^3] [^3]

For a dengue SEIR–SEI model, the corresponding implementation would have the network output $$[S_h,E_h,I_h,R_h,S_m,E_m,I_m]$$, use a reporting operator on human incidence or cases, and place the residual of all seven equations in the physics term. If mosquito infection data are absent, $$I_m$$ is inferred—not measured—and its uncertainty should be assessed with multiple initializations, posterior or ensemble methods, sensitivity/observability analysis, and independent mosquito-surveillance validation. The delivered table identifies which papers actually address each of those components and which remain mainly synthetic parameter-discovery demonstrations.

[^1]: Rodr'iguez et al., 2022. EINNs: Epidemiologically-Informed Neural Networks. AAAI Conference on Artificial Intelligence.

[^2]: Kharazmi et al., 2021. Identifiability and predictability of integer- and fractional-order epidemiological models using physics-informed neural networks. Nature Computational Science.

[^3]: Komatsu, 2024. Estimate Epidemiological Parameters given Partial Observations based on Algebraically Observable PINNs. arXiv.org.

[^4]: Schiassi et al., 2021. Physics-Informed Neural Networks and Functional Interpolation for Data-Driven Parameters Discovery of Epidemiological Compartmental Models. Mathematics.