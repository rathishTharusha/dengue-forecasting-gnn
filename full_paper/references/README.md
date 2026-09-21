# Primary Literature References

This folder contains the complete reference papers cited in the CS3631 Phase 2 short paper (*"Know When the Epidemic Comes: Mathematics Before Data in Dengue Outbreak Forecasting"*).

## Reference Papers Index

| Key | Title | Authors & Venue | PDF Document |
| :--- | :--- | :--- | :--- |
| `tissera2020severe` | **Severe Dengue Epidemic, Sri Lanka, 2017** | Tissera et al., *Emerging Infectious Diseases* (CDC), 2020 | [`Tissera_2020_Severe_Dengue_Sri_Lanka_2017.pdf`](file:///short_paper/references/Tissera_2020_Severe_Dengue_Sri_Lanka_2017.pdf) |
| `weng2024graph` | **Graph Representation Learning for Dengue Forecasting** | Weng et al., *IEEE BigData*, 2024 | [`Weng_2024_Graph_Representation_Learning_Dengue.pdf`](file:///short_paper/references/Weng_2024_Graph_Representation_Learning_Dengue.pdf) |
| `gulmohamed2026denguegnn` | **DengueGNN: Graph-based deep learning for modeling disease spread dynamics and prediction** | GulMohamed et al., *Scientific Reports*, 2026 | [`GulMohamed_2026_DengueGNN.pdf`](file:///short_paper/references/GulMohamed_2026_DengueGNN.pdf) |
| `phaijoo2018sensitivity` | **Sensitivity Analysis of SEIR-SEI Model of Dengue Disease** | Phaijoo & Gurung, *GAMS J. Math. & Math. Biosci.*, 2018 | [`Phaijoo_Gurung_2018_Sensitivity_Analysis_SEIR_SEI.pdf`](file:///short_paper/references/Phaijoo_Gurung_2018_Sensitivity_Analysis_SEIR_SEI.pdf) |
| `seirsei2022` | **Analysis of Vector-host SEIR-SEI Dengue Epidemiological Model** | Hasan et al., *Int. J. Analysis & Applications*, 2022 | [`Hasan_2022_SEIR_SEI_Dengue.pdf`](file:///short_paper/references/Hasan_2022_SEIR_SEI_Dengue.pdf) |
| `raissi2019pinn` | **Physics-Informed Neural Networks** | Raissi, Perdikaris, Karniadakis, *J. Comp. Physics*, 2019 | [`Raissi_2019_Physics_Informed_Neural_Networks.pdf`](file:///short_paper/references/Raissi_2019_Physics_Informed_Neural_Networks.pdf) |
| `nguyen2024mppinn` | **MP-PINN: A Multi-Phase Physics-Informed Neural Network for Epidemic Forecasting** | Nguyen et al., *arXiv:2411.06781*, 2024 | [`Nguyen_2024_MP_PINN_Epidemic_Forecasting.pdf`](file:///short_paper/references/Nguyen_2024_MP_PINN_Epidemic_Forecasting.pdf) |
| `wu2019graphwavenet` | **Graph WaveNet for Deep Spatial-Temporal Graph Modeling** | Wu et al., *IJCAI*, 2019 | [`Wu_2019_Graph_WaveNet.pdf`](file:///short_paper/references/Wu_2019_Graph_WaveNet.pdf) |
| `li2018dcrnn` | **Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting** | Li et al., *ICLR*, 2018 | [`Li_2018_DCRNN_Traffic_Forecasting.pdf`](file:///short_paper/references/Li_2018_DCRNN_Traffic_Forecasting.pdf) |
| `guo2019astgcn` | **Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting** | Guo et al., *AAAI*, 2019 | [`Guo_2019_ASTGCN.pdf`](file:///short_paper/references/Guo_2019_ASTGCN.pdf) |
| `bai2021a3tgcn` | **A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting** | Bai et al., *ISPRS Int. J. Geo-Inf.*, 2021 | [`Bai_2021_A3TGCN.pdf`](file:///short_paper/references/Bai_2021_A3TGCN.pdf) |
| `shi2019aagcn` | **Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition** | Shi et al., *CVPR*, 2019 | [`Shi_2019_AAGCN.pdf`](file:///short_paper/references/Shi_2019_AAGCN.pdf) |
| `wallinga2007generation` | **How Generation Intervals Shape the Relationship Between Growth Rates and Reproductive Numbers** | Wallinga & Lipsitch, *Proc. R. Soc. B*, 2007 | DOI: `10.1098/rspb.2006.3754` / PMC1766383 |

---

## Citation Integrity

Every statement in the short paper corresponds to one of three categories:
1. **Mathematical / Common Truth**: e.g., Jensen's inequality $\mathbb{E}[\sqrt{X}] \le \sqrt{\mathbb{E}[X]}$, lag-1 persistence $\hat{y}_{t+h}=y_t$, standard graph Laplacian definitions.
2. **Empirically Proved in Our Research**:
   - Week-395 reporting backlog (19x spike in 18/25 districts, carrying ~90% of fold error).
   - Exact reproduction of 5 ST-GNN baselines within 7.3% and proof of training-inclusive per-window reporting.
   - Dual-scoring persistence floor (44.80 all-windows vs 29.52 artifact-free).
   - Predictability ceiling of $R_t$ ($r^2=0.263$, compounding $1.90\times$ multiplicative error).
   - Relaxed biological envelope ($r_{\text{max}} = 2.3884$) and normalized spatial Dirichlet smoothness ($-0.044$ RMSE, 57/60 paired runs, $p<0.0001$).
   - Outbreak timing detection (AUC $0.807 \to 0.826$).
3. **Backed by Peer-Reviewed External Research**:
   - Sri Lankan Dengue Epidemic Statistics $\to$ Tissera et al. (2020)
   - Baseline Dataset & Graph $\to$ Weng et al. (2024)
   - Increments & Multi-Horizon Heads $\to$ GulMohamed et al. (2026)
   - SEIR-SEI Compartmental Stage Durations $\to$ Phaijoo & Gurung (2018), Hasan et al. (2022)
   - PINN Formulations $\to$ Raissi et al. (2019), Nguyen et al. (2024)
   - ST-GNN Architectures $\to$ Li et al. (2018), Guo et al. (2019), Wu et al. (2019), Shi et al. (2019), Bai et al. (2021)
   - Generation Interval Dynamics $\to$ Wallinga & Lipsitch (2007)
