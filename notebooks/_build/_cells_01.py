from gen_notebooks import md, code

CELLS = []
C = CELLS

C.append(md(r"""
# 01 — Classical / Traditional Baselines (Weng et al. Table I)

Reproduces the **traditional EWS baselines** from Weng et al. (2024): ARIMAX, Random
Forest, XGBoost, ARNN, and LSTM — each fit **per district** (25 Sri Lanka districts),
evaluated with the paper's **rolling-origin protocol**: 5 growing segments
(60/70/80/90/100% of the series), each split 70/30 train/test, MAE & RMSE averaged
across segments.

We also add two baselines the literature review flags as standard-but-missing
(Area 5, "Baselines to include"): **naive persistence** and **seasonal-naive**. These
cost nothing to compute and give a lower bound — if a model can't beat seasonal-naive,
that's worth knowing.

**Requires:** `data_manifest.json` from `00_data_setup_eda.ipynb` (run that first).

**Runtime note:** the full reproduction (25 districts × 5 segments × 5-7 models) takes a
while, mostly from ARIMAX and LSTM. Use `QUICK_TEST = True` first to check everything
runs end-to-end on a handful of districts/segments, then flip to `False` for the real run.
"""))

C.append(code(r"""
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "statsmodels", "scikit-learn", "xgboost", "tqdm"])

import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
np.random.seed(0)
"""))

C.append(code(r"""
COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
}
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, BASELINE = "#e1e0d9", "#fcfcfb", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 110,
})

def style_ax(ax):
    ax.grid(axis="y", linewidth=0.8, color=GRID)
    ax.set_axisbelow(True)
    return ax
"""))

C.append(md(r"""## Config"""))

C.append(code(r"""
QUICK_TEST = True   # <-- set False for the real, full reproduction before writing up results

manifest = json.load(open("./data_manifest.json"))
df = pd.read_csv(manifest["csv_path"])
df["Week"] = pd.to_datetime(df["Week"])

DISEASE_ONLY = manifest["n_features"] == 1  # True until NASA covariates are joined in

ALL_DISTRICTS = manifest["districts"]
DISTRICTS = ALL_DISTRICTS[:4] if QUICK_TEST else ALL_DISTRICTS
SEGMENTS = [1.0] if QUICK_TEST else [0.6, 0.7, 0.8, 0.9, 1.0]
TRAIN_FRAC = 0.7  # matches the reference repo's 70/30 split within each segment

RESULTS_DIR = "./results"
import os
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"QUICK_TEST={QUICK_TEST} | districts={len(DISTRICTS)} | segments={SEGMENTS} | "
      f"DISEASE_ONLY={DISEASE_ONLY} (mode={manifest['mode']!r}, n_features={manifest['n_features']})")
if DISEASE_ONLY:
    print("No meteorological covariates available yet — ARIMA runs without exog, "
          "and Random Forest/XGBoost use engineered lag features instead of weather data.")
"""))

C.append(code(r"""
# Reference numbers from Weng et al. (2024) Table I ("Shifted Dataset", cross-validated
# column) and the repo's results.txt — used later purely as a sanity check, not ground
# truth we're required to match exactly (different train seeds / library versions will
# shift results somewhat).
REFERENCE_CV_SHIFTED = {
    "ARIMA":         {"MAE": 168.37, "RMSE": 189.22},
    "Random Forest": {"MAE": 48.91,  "RMSE": 84.66},
    "XGBoost":       {"MAE": 53.89,  "RMSE": 95.22},
    "ARNN":          {"MAE": 82.46,  "RMSE": 106.83},
    "LSTM":          {"MAE": 81.19,  "RMSE": 131.36},
}
"""))

C.append(md(r"""
## Per-district feature/target split

Two modes, chosen automatically from `DISEASE_ONLY` (set by notebook 00's manifest):

- **Processed mode** (meteorological covariates available): mirrors the reference
  repo's `arima.py` / `random_forest.py` / `lstm.py` — `cases` is the target; the
  dropped columns are identifiers, redundant min/max variants, or covariates the
  original team excluded for these models.
- **Disease-only mode** (current data): there are no covariate columns to drop, so
  Random Forest/XGBoost instead get **engineered lag features** (previous 1/2/3 weeks'
  case counts + a trailing 4-week rolling mean) — the standard way to give a
  non-sequential model like a tree ensemble access to recent history. ARIMA does **not**
  get these as exog (see the ARIMA cell below) — its AR terms already model
  autocorrelation, and feeding it lagged-y as exog is redundant/can destabilize fitting.
"""))

C.append(code(r"""
DROP_COLS = [
    "Week", "region", "cases", "minTime", "minNdvi", "maxNdvi", "meanNdvi",
    "minPrecipitationcal", "maxPrecipitationcal", "meanPrecipitationcal",
    "meanCanopint_Inst", "meanPsurf_F_Inst",
]

def get_district_xy(df, district):
    sub = df[df["region"] == district].sort_values("Week").reset_index(drop=True)
    y_full = sub["cases"].astype(float)

    if DISEASE_ONLY:
        X_full = pd.DataFrame({
            "lag1": y_full.shift(1),
            "lag2": y_full.shift(2),
            "lag3": y_full.shift(3),
            "roll_mean_4": y_full.shift(1).rolling(4).mean(),
        })
        valid = X_full.notna().all(axis=1)
        y = y_full[valid].reset_index(drop=True)
        X = X_full[valid].reset_index(drop=True)
        weeks = sub["Week"][valid].reset_index(drop=True)
        missing = []
    else:
        y, weeks = y_full, sub["Week"]
        present = [c for c in DROP_COLS if c in sub.columns]
        missing = [c for c in DROP_COLS if c not in sub.columns]
        X = sub.drop(columns=present).select_dtypes(include=[np.number]).fillna(0.0)

    return y, X, weeks, missing

_, _example_X, _, _missing = get_district_xy(df, DISTRICTS[0])
if _missing:
    print(f"Note: expected columns not found in this CSV (skipped): {_missing}")
print(f"Feature set for RF/XGBoost ({_example_X.shape[1]} cols):", list(_example_X.columns))
"""))

C.append(md(r"""
## Model fit/predict functions

Each takes `(train_y, test_y, train_X, test_X)` and returns `(truth, prediction)` as
numpy arrays, so one harness can drive all of them uniformly.
"""))

C.append(code(r"""
def fit_predict_naive(train_y, test_y, train_X, test_X):
    last = train_y.values[-1]
    return test_y.values, np.full(len(test_y), last)


def fit_predict_seasonal_naive(train_y, test_y, train_X, test_X, season=52):
    combined = pd.concat([train_y, test_y]).reset_index(drop=True)
    n_train = len(train_y)
    preds = [
        combined.iloc[n_train + i - season] if (n_train + i - season) >= 0 else train_y.values[-1]
        for i in range(len(test_y))
    ]
    return test_y.values, np.array(preds)


def fit_predict_arimax(train_y, test_y, train_X, test_X):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    exog_train = None if DISEASE_ONLY else train_X
    exog_test = None if DISEASE_ONLY else test_X
    model = SARIMAX(train_y, exog=exog_train, order=(1, 2, 3), seasonal_order=(0, 0, 0, 0))
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(test_y), exog=exog_test)
    return test_y.values, forecast.to_numpy()


def fit_predict_rf(train_y, test_y, train_X, test_X):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(train_X, train_y)
    return test_y.values, m.predict(test_X)


def fit_predict_xgb(train_y, test_y, train_X, test_X):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    m.fit(train_X, train_y)
    return test_y.values, m.predict(test_X)


NON_WINDOWED_MODELS = {
    "Naive": fit_predict_naive,
    "Seasonal-Naive": fit_predict_seasonal_naive,
    "ARIMA": fit_predict_arimax,
    "Random Forest": fit_predict_rf,
    "XGBoost": fit_predict_xgb,
}
"""))

C.append(md(r"""## Rolling-origin harness (naive, seasonal-naive, ARIMAX, RF, XGBoost)"""))

C.append(code(r"""
def mae_rmse(truth, pred):
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    mae = np.mean(np.abs(truth - pred))
    rmse = np.sqrt(np.mean((truth - pred) ** 2))
    return mae, rmse


rows = []
for district in tqdm(DISTRICTS, desc="districts"):
    y, X, weeks, _ = get_district_xy(df, district)
    for seg in SEGMENTS:
        n = int(len(y) * seg)
        y_s, X_s = y.iloc[:n].reset_index(drop=True), X.iloc[:n].reset_index(drop=True)
        split = int(len(y_s) * TRAIN_FRAC)
        train_y, test_y = y_s.iloc[:split], y_s.iloc[split:]
        train_X, test_X = X_s.iloc[:split], X_s.iloc[split:]
        if len(test_y) < 2:
            continue
        for model_name, fn in NON_WINDOWED_MODELS.items():
            try:
                truth, pred = fn(train_y, test_y, train_X, test_X)
                mae, rmse = mae_rmse(truth, pred)
            except Exception as e:
                mae, rmse = np.nan, np.nan
                print(f"  [{district} | seg={seg} | {model_name}] failed: {e}")
            rows.append({"district": district, "segment": seg, "model": model_name,
                         "mae": mae, "rmse": rmse})

results_classical = pd.DataFrame(rows)
results_classical.to_csv(f"{RESULTS_DIR}/baseline_classical_nonwindowed.csv", index=False)
results_classical.groupby("model")[["mae", "rmse"]].mean()
"""))

C.append(md(r"""
## LSTM (windowed, per district)

Follows the reference repo's `lstm.py`: `MinMaxScaler` on X and y, 3-week input window,
single-layer LSTM(50) + Dense(1), 40 epochs, batch size 4, predicting 1 step ahead.
"""))

C.append(code(r"""
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow"])
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

WINDOW = 3
LSTM_EPOCHS = 10 if QUICK_TEST else 40

def make_windows(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


rows = []
for district in tqdm(DISTRICTS, desc="districts (LSTM)"):
    # In disease-only mode, get_district_xy's X is already lag-engineered (for RF/XGBoost) —
    # windowing that again would window-of-windows. Instead pull the raw case series here
    # and let the LSTM do its own windowing directly on cases (a standard univariate setup).
    sub = df[df["region"] == district].sort_values("Week").reset_index(drop=True)
    y_raw, weeks_raw = sub["cases"].astype(float), sub["Week"]

    for seg in SEGMENTS:
        n = int(len(y_raw) * seg)
        y_s = y_raw.iloc[:n].reset_index(drop=True)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y_s.values.reshape(-1, 1))

        if DISEASE_ONLY:
            X_scaled = y_scaled  # univariate: window of past cases predicts next
        else:
            _, X, _, _ = get_district_xy(df, district)
            X_s = X.iloc[:n].reset_index(drop=True)
            scaler_x = MinMaxScaler()
            X_scaled = scaler_x.fit_transform(X_s)

        X_seq, y_seq = make_windows(X_scaled, y_scaled, WINDOW)
        if len(X_seq) < 10:
            continue
        split = int(len(X_seq) * TRAIN_FRAC)
        train_X, test_X = X_seq[:split], X_seq[split:]
        train_y, test_y = y_seq[:split], y_seq[split:]

        try:
            model = keras.Sequential([
                layers.LSTM(50, activation="relu", input_shape=(train_X.shape[1], train_X.shape[2])),
                layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(train_X, train_y, epochs=LSTM_EPOCHS, batch_size=4, verbose=0, shuffle=False)

            pred_scaled = model.predict(test_X, verbose=0)
            pred = scaler_y.inverse_transform(pred_scaled).flatten()
            truth = scaler_y.inverse_transform(test_y).flatten()
            mae, rmse = mae_rmse(truth, pred)
        except Exception as e:
            mae, rmse = np.nan, np.nan
            print(f"  [{district} | seg={seg} | LSTM] failed: {e}")

        rows.append({"district": district, "segment": seg, "model": "LSTM", "mae": mae, "rmse": rmse})

results_lstm = pd.DataFrame(rows)
results_lstm.to_csv(f"{RESULTS_DIR}/baseline_classical_lstm.csv", index=False)
results_lstm[["mae", "rmse"]].mean()
"""))

C.append(md(r"""
## ARNN (optional — NeuralProphet)

The reference repo trains ARNN "out-of-the-box using NeuralProphet." This library has
heavier dependencies and slower per-fit time than the models above, so it's gated behind
`RUN_ARNN` and — even outside `QUICK_TEST` — capped to a handful of districts by default.
Bump `ARNN_DISTRICTS` once you've confirmed it runs.
"""))

C.append(code(r"""
RUN_ARNN = True
ARNN_DISTRICTS = DISTRICTS[:2] if QUICK_TEST else DISTRICTS[:6]
ARNN_EPOCHS = 20 if QUICK_TEST else 50

rows = []
if RUN_ARNN:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralprophet"])
    try:
        from neuralprophet import NeuralProphet, set_log_level
        set_log_level("ERROR")

        for district in tqdm(ARNN_DISTRICTS, desc="districts (ARNN)"):
            y, X, weeks, _ = get_district_xy(df, district)
            for seg in SEGMENTS:
                n = int(len(y) * seg)
                y_s, weeks_s = y.iloc[:n].reset_index(drop=True), weeks.iloc[:n].reset_index(drop=True)
                split = int(len(y_s) * TRAIN_FRAC)
                train_df = pd.DataFrame({"ds": weeks_s.iloc[:split], "y": y_s.iloc[:split].values})
                test_y = y_s.iloc[split:].values
                if len(test_y) < 2:
                    continue
                try:
                    m = NeuralProphet(n_lags=3, n_forecasts=1, epochs=ARNN_EPOCHS, learning_rate=0.01)
                    m.fit(train_df, freq="W")
                    future = m.make_future_dataframe(train_df, periods=len(test_y), n_historic_predictions=False)
                    forecast = m.predict(future)
                    pred = forecast["yhat1"].values[: len(test_y)]
                    mae, rmse = mae_rmse(test_y[: len(pred)], pred)
                except Exception as e:
                    mae, rmse = np.nan, np.nan
                    print(f"  [{district} | seg={seg} | ARNN] failed: {e}")
                rows.append({"district": district, "segment": seg, "model": "ARNN", "mae": mae, "rmse": rmse})
    except ImportError:
        print("neuralprophet unavailable — skipping ARNN. Set RUN_ARNN=False to silence this.")

results_arnn = pd.DataFrame(rows)
if len(results_arnn):
    results_arnn.to_csv(f"{RESULTS_DIR}/baseline_classical_arnn.csv", index=False)
results_arnn[["mae", "rmse"]].mean() if len(results_arnn) else "No ARNN results."
"""))

C.append(md(r"""## Combine, aggregate, and compare against the paper"""))

C.append(code(r"""
all_results = pd.concat([results_classical, results_lstm, results_arnn], ignore_index=True)
all_results.to_csv(f"{RESULTS_DIR}/baseline_classical_all.csv", index=False)

leaderboard = (
    all_results.groupby("model")[["mae", "rmse"]]
    .agg(["mean", "std"])
    .round(2)
    .sort_values(("rmse", "mean"))
)
leaderboard
"""))

C.append(code(r"""
print(f"{'Model':<16} {'Our MAE':>10} {'Paper MAE':>10} | {'Our RMSE':>10} {'Paper RMSE':>10}")
for model in leaderboard.index:
    our_mae = leaderboard.loc[model, ("mae", "mean")]
    our_rmse = leaderboard.loc[model, ("rmse", "mean")]
    ref = REFERENCE_CV_SHIFTED.get(model)
    ref_mae = ref["MAE"] if ref else float("nan")
    ref_rmse = ref["RMSE"] if ref else float("nan")
    print(f"{model:<16} {our_mae:>10.2f} {ref_mae:>10.2f} | {our_rmse:>10.2f} {ref_rmse:>10.2f}")

print("\nNote: exact parity isn't expected — QUICK_TEST subsets districts/segments, library/\n"
      "version drift shifts fits slightly, and (until covariates are joined in) this run is\n"
      "disease-only on a different date range than the paper's, so these reference numbers\n"
      "are a rough sanity check, not a target to match. What matters is the *relative\n"
      "ordering*: GNNs (notebook 02) should beat these; among these, Random Forest/XGBoost\n"
      "should beat plain ARIMA/LSTM on the lag-feature set.")
if DISEASE_ONLY:
    print("\n(DISEASE_ONLY=True — the reference numbers above were fit WITH meteorological "
          "covariates, ours without. Treat this comparison as directional only.)")
"""))

C.append(md(r"""## Plots"""))

C.append(code(r"""
# Leaderboard: models are ranked by magnitude (RMSE), not identity — so a single accent
# hue for the paper's own baselines and a second for our added naive/seasonal-naive
# extensions communicates "which of these did Weng et al. report" without a 7-color legend.
is_extension = leaderboard.index.isin(["Naive", "Seasonal-Naive"])
bar_colors = [COLORS["orange"] if ext else COLORS["blue"] for ext in is_extension]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(leaderboard))
ax.barh(y_pos, leaderboard[("rmse", "mean")], color=bar_colors, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(leaderboard.index)
ax.invert_yaxis()
style_ax(ax)
ax.set_xlabel("RMSE (lower is better)")
ax.set_title("Classical baselines — mean RMSE across districts & segments", loc="left", fontsize=13)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLORS["blue"], label="Weng et al. (2024) baseline"),
    Patch(color=COLORS["orange"], label="Added extension (naive/seasonal-naive)"),
], frameon=False, loc="lower right")
plt.tight_layout()
plt.show()
"""))

C.append(code(r"""
# Actual-vs-predicted for one district with the best-performing model (paper's Fig. 5 style)
best_model = leaderboard.index[0]
demo_district = "Colombo" if "Colombo" in DISTRICTS else DISTRICTS[0]
y, X, weeks, _ = get_district_xy(df, demo_district)
split = int(len(y) * TRAIN_FRAC)
train_y, test_y = y.iloc[:split], y.iloc[split:]
train_X, test_X = X.iloc[:split], X.iloc[split:]

fn = NON_WINDOWED_MODELS.get(best_model)
if fn is not None:
    truth, pred = fn(train_y, test_y, train_X, test_X)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weeks.iloc[:split], train_y, color=INK_MUTED, linewidth=1.2, label="Train (actual)")
    ax.plot(weeks.iloc[split:], truth, color=COLORS["blue"], linewidth=1.6, label="Test (actual)")
    ax.plot(weeks.iloc[split:], pred, color=COLORS["orange"], linewidth=1.6,
            linestyle="--", label=f"{best_model} (predicted)")
    style_ax(ax)
    ax.set_title(f"{best_model} — {demo_district}", loc="left", fontsize=13)
    ax.set_ylabel("Dengue cases")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()
else:
    print(f"'{best_model}' is windowed (LSTM/ARNN) — skipping this quick-look plot; "
          "see the training loop above for its own forecast traces.")
"""))

C.append(md(r"""
## Next steps

- **`02_baselines_gnn.ipynb`** — the spatio-temporal GNNs (STGAT, A3TGCN, ASTGCN, DCRNN,
  AAGCN) that Weng et al. show consistently beating everything in this notebook. That
  comparison (traditional vs. graph-aware) is the empirical core of the baseline
  reproduction milestone.
- Once both notebooks' `results/baseline_*_all.csv` exist, the final cell of notebook 02
  merges them into one combined leaderboard — that combined table is what belongs in the
  "Baseline Reproduction" section of the project proposal.
"""))
