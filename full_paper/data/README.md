# Dataset Sources, Acquisition Links, & Processing Pipeline

This folder contains all cleaned, processed, and aligned datasets required to run and reproduce the **Physics-Informed SEIR-GNN Dengue Outbreak Forecasting** experiments.

---

## 🌐 1. Primary Data Sources & Download Links

| Dataset | Provider / Source | Access URL / Portal | Resolution / Format |
|:---|:---|:---|:---|
| **Epidemiological Case Series** | Epidemiology Unit, Ministry of Health, Sri Lanka | [http://www.epid.gov.lk/web/](http://www.epid.gov.lk/web/) | Weekly reported dengue cases across 25 administrative districts (May 2013 – Feb 2024; 559 weeks). |
| **ERA5 Climate Reanalysis** | ECMWF / Copernicus Climate Change Service (C3S) | [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) | Weekly aggregated 2m Temperature ($K$), Total Precipitation ($m$), Dewpoint, Pressure, Wind components. |
| **MODIS Vegetation Index (NDVI)** | NASA LP DAAC / USGS EROS (MOD13Q1 / MYD13Q1) | [https://lpdaac.usgs.gov/products/mod13q1v061/](https://lpdaac.usgs.gov/products/mod13q1v061/) | 16-day 250m Normalized Difference Vegetation Index (NDVI) interpolated to weekly district means. |
| **District Population Metadata** | Department of Census and Statistics (DCS), Sri Lanka | [http://www.statistics.gov.lk/](http://www.statistics.gov.lk/) | Official Census 2012 district population totals ($N_i$) & annual growth projections. |
| **Field Seroprevalence Data** | Tissera et al. (2020) Field Survey / MOH Sri Lanka | Peer-reviewed empirical study (Tissera et al., 2020) | Seropositive immune fraction ($1 - S/N$) across 9 key districts. |

---

## 🛠️ 2. Data Processing & Alignment Workflow

### Step 1: Epidemiological Case Series Cleaning (`dengue_cases_raw.csv` & `rebuilt_index.csv`)
- **Extraction**: Weekly Epidemiological Reports (WER) published by the Ministry of Health were digitized and parsed into a 25-district time series spanning May 2013 to February 2024 (559 consecutive weeks).
- **Missing Week Handling**: Missing surveillance weeks (7 weeks total) were linearly interpolated using adjacent district observations.
- **Calendar Alignment**: Standardized to Monday-starting ISO epidemiological calendar weeks (`calendar_index.csv`).

### Step 2: Climate Feature Extraction (`era5_weekly_by_district.csv`)
- **Extraction**: Gridded hourly ERA5-Land reanalysis rasters were spatially averaged over the spatial boundaries of each of the 25 Sri Lankan administrative districts using district shapefile geometries.
- **Features Extracted**:
  - `t2m`: 2m Mean Air Temperature (Kelvin)
  - `tp`: Total Weekly Cumulative Precipitation (meters)
  - `d2m`: Dewpoint Temperature (Kelvin)
  - `sp`: Surface Pressure ($Pa$)
  - `u10`, `v10`: 10m Wind Components ($m/s$)

### Step 3: Vegetation Dynamics (`modis_ndvi_weekly_by_district.csv`)
- **Extraction**: MODIS 16-day composite 250m NDVI surface reflectances were aggregated over district polygons and cubic-spline interpolated to match weekly epidemiological timesteps.

### Step 4: Spatial Graph Construction & SEIR Parameterization (`seir_parameters.json`)
- **Spatial Adjacency Matrix ($A_{ij}$)**: Constructed based on shared geographic district borders (Contiguity Matrix) and normalized by degree $\hat{A}_{ij} = A_{ij} / \sum_k A_{ik}$.
- **SEIR Compartmental Parameters**: Extracted from Phaijoo & Gurung (2018) for Sri Lankan dengue dynamics:
  - Extrinsic Incubation Rate: $\nu_v = 1/10 \text{ day}^{-1}$
  - Intrinsic Incubation Rate: $\nu_h = 1/6 \text{ day}^{-1}$ ($\omega = 0.1667 \text{ day}^{-1}$)
  - Infectious Period Recovery Rate: $\gamma_h = 1/7 \text{ day}^{-1}$ ($\gamma = 0.1428 \text{ day}^{-1}$)

---

## 📁 3. Local Dataset Files in `full_paper/data/`

* `dengue_cases_raw.csv`: Raw weekly epidemiological case counts for 25 Sri Lankan districts.
* `era5_weekly_by_district.csv`: Aligned weekly climate reanalysis variables per district.
* `modis_ndvi_weekly_by_district.csv`: Weekly satellite NDVI vegetation index values per district.
* `district_census_2012.csv`: Official district population totals and land areas.
* `seroprevalence_nine_districts.csv`: Seroprevalence field survey measurements for 9 districts.
* `seir_parameters.json`: System parameters for the SEIR compartmental model.
