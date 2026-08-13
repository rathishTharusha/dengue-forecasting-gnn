from gen_notebooks import md, code

CELLS = []
C = CELLS

C.append(md(r"""
# 02 — Spatio-Temporal GNN Baselines (Weng et al. Table I)

Reproduces the **graph-based EWS models** from Weng et al. (2024): **STGAT, A3TGCN,
ASTGCN, DCRNN, AAGCN**, trained via
[PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/) on the
organic district-adjacency graph, exactly as in the reference implementation
([`disease_modeling_MLOS2/Models`](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2/tree/main/Models)).

These are the models the paper shows **consistently beating every traditional baseline**
in `01_baselines_classical.ipynb` — reproducing that gap is the empirical center of the
"baseline reproduction" milestone, and the jumping-off point for the project's actual
novelty (physics-informed loss, then GAN augmentation, per the literature review).

**Requires:** `data_manifest.json` from `00_data_setup_eda.ipynb`.

**Use a GPU runtime if you can** (Colab: *Runtime → Change runtime type → T4 GPU* /
Kaggle: enable an accelerator) — training 5 models × up to 5 rolling segments × 50
epochs is slow on CPU. The reference repo itself trained on CPU, so `DEVICE="cpu"`
still reproduces their numbers if no GPU is available; it's just slower.
"""))

C.append(md(r"""## 1. Install PyG + PyG-Temporal (version-matched to the reference repo)"""))

C.append(code(r"""
import sys, subprocess, importlib

def sh(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)

try:
    import torch
except ImportError:
    raise RuntimeError("This notebook needs a PyTorch runtime (Colab/Kaggle both provide one).")

TORCH_VER = torch.__version__.split("+")[0]
HAS_CUDA = torch.cuda.is_available()
CU_TAG = ("cu" + torch.version.cuda.replace(".", "")) if (HAS_CUDA and torch.version.cuda) else "cpu"
WHEEL_INDEX = f"https://data.pyg.org/whl/torch-{TORCH_VER}+{CU_TAG}.html"

print(f"torch {TORCH_VER} | cuda available: {HAS_CUDA} | wheel index: {WHEEL_INDEX}")

try:
    import torch_geometric, torch_geometric_temporal
except ImportError:
    sh("install", "-q", "torch_geometric")
    sh("install", "-q", "torch_geometric_temporal==0.54.0")
    try:
        sh("install", "-q", "torch_scatter", "torch_sparse", "-f", WHEEL_INDEX)
    except subprocess.CalledProcessError:
        print("torch_scatter/torch_sparse wheel install failed — most layers used here "
              "(GATConv, ChebConv-based DCRNN) don't need them, so continuing.")
    importlib.invalidate_caches()
    import torch_geometric, torch_geometric_temporal

print("torch_geometric", torch_geometric.__version__)
print("torch_geometric_temporal", torch_geometric_temporal.__version__)

DEVICE = "cuda" if HAS_CUDA else "cpu"
print("Using device:", DEVICE)
"""))

C.append(md(r"""
## 2. Get the model architectures (`gnn_models.py`)

Rather than re-transcribing STGAT/A3TGCN/ASTGCN/DCRNN/AAGCN by hand (and risking a subtle
mismatch from the paper), we clone the reference repo and import its `gnn_models.py`
directly — same architecture, same hyperparameter defaults (2 ASTGCN blocks, 64 filters,
K=3 Chebyshev order for DCRNN, etc).
"""))

C.append(code(r"""
REPO_DIR = "disease_modeling_MLOS2"
if not __import__("os").path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                     "https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2"], check=True)

sys.path.insert(0, f"{REPO_DIR}/Models")
from gnn_models import STGAT, TemporalGCN, AttentionSTGCN, DConvRNN, AdaptiveGCN
print("Model classes imported OK.")
"""))

C.append(md(r"""## 3. Chart style (same palette as notebooks 00/01)"""))

C.append(code(r"""
import matplotlib.pyplot as plt

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

C.append(md(r"""## 4. Config & data paths"""))

C.append(code(r"""
import json, random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

QUICK_TEST = True  # <-- set False for the real, full reproduction before writing up results

manifest = json.load(open("./data_manifest.json"))
DATA_FILE = manifest["npy_path"]
ADJ_PATH = manifest["adj_path"]
N_FEATURES = manifest["n_features"]
DISEASE_FEATURE_INDEX = manifest["disease_feature_index"]
DISEASE_ONLY = N_FEATURES == 1  # True until NASA covariates are joined in; forces every
                                 # model below into its single-feature code path

SEGMENTS = [1.0] if QUICK_TEST else [0.6, 0.7, 0.8, 0.9, 1.0]
num_epochs = 5 if QUICK_TEST else 50   # reference repo uses 50

# Hyperparameters — unchanged from the reference repo's evaluation.py
window_size = 3
in_channels, out_channels, num_nodes = 3, 3, 25
lr, decay, dropout = 1e-4, 5e-5, 0.1

index = {i: v for i, v in enumerate(sorted(json.load(open(ADJ_PATH)).keys()))}
print(f"QUICK_TEST={QUICK_TEST} | epochs={num_epochs} | segments={SEGMENTS} | device={DEVICE}")
print(f"data file: {DATA_FILE} (N_FEATURES={N_FEATURES}, DISEASE_ONLY={DISEASE_ONLY})")
print(f"adjacency: {ADJ_PATH}")
"""))

C.append(code(r"""
# Reference numbers — Weng et al. (2024) Table I, "Shifted Dataset" / cross-validated column.
REFERENCE_CV_SHIFTED = {
    "ARIMA":         {"MAE": 168.37, "RMSE": 189.22},
    "Random Forest": {"MAE": 48.91,  "RMSE": 84.66},
    "XGBoost":       {"MAE": 53.89,  "RMSE": 95.22},
    "ARNN":          {"MAE": 82.46,  "RMSE": 106.83},
    "LSTM":          {"MAE": 81.19,  "RMSE": 131.36},
    "DCRNN":         {"MAE": 45.98,  "RMSE": 71.61},
    "STGAT":         {"MAE": 25.38,  "RMSE": 44.78},
    "AAGCN":         {"MAE": 41.93,  "RMSE": 55.83},
    "A3TGCN":        {"MAE": 34.03,  "RMSE": 58.52},
    "ASTGCN":        {"MAE": 33.68,  "RMSE": 47.72},
}
"""))

C.append(md(r"""
## 5. Data & graph utilities

Transcribed from the reference repo's `evaluation.py` — same z-normalization, same
windowing (3-week input → 3-week-ahead target), same edge-index construction from the
adjacency list.
"""))

C.append(code(r"""
def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_pred - y_true)) / (y_true + 1e-15) * 100)

def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))

def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def z_norm(x, mean, std):
    return (x - mean) / std

def inverse_z_norm(x, mean, std):
    return x * std + mean

def load_adjacency_matrix(adj_file, self_loop):
    adj = json.load(open(adj_file))
    n = len(adj.keys())
    idx_map = {v: i for i, v in enumerate(sorted(adj.keys()))}
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    if self_loop:
        for i in range(n):
            adj_matrix[i][i] = 1
    for district in adj:
        for nei in adj[district]:
            adj_matrix[idx_map[district]][idx_map[nei]] = 1
    return adj_matrix

def load_data(data_file, use_disease_only, subset=1.0):
    x = np.nan_to_num(np.load(data_file, allow_pickle=True))
    if not use_disease_only:
        mean, std = np.mean(x), np.std(x)
    else:
        mean, std = np.mean(x[..., DISEASE_FEATURE_INDEX]), np.std(x[..., DISEASE_FEATURE_INDEX])
    x = z_norm(torch.tensor(x), mean, std)
    x = x[: int(x.shape[0] * subset), ...]
    return x, mean, std
"""))

C.append(code(r"""
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split


def create_dataset_single_shot(window_size=3, predict_ahead=3, self_loop=True,
                                train_split=0.7, valid_split=0.1,
                                use_disease_only=False, subset=1.0):
    # Returns temporal signal of shape Nodes x Feats x WindowSize (for A3TGCN).
    adj = load_adjacency_matrix(ADJ_PATH, self_loop)
    edge_index = torch.tensor(
        [[x, y] for x in range(25) for y in range(25) if adj[x][y]], dtype=torch.long
    )
    x, mean, std = load_data(DATA_FILE, use_disease_only, subset)

    features, targets = [], []
    for i in range(window_size, x.shape[0] - predict_ahead):
        if use_disease_only:
            x_data = np.einsum("ij->ji", x[i - window_size : i, :, DISEASE_FEATURE_INDEX])
            y_data = np.einsum("ij->ji", x[i : i + predict_ahead, :, DISEASE_FEATURE_INDEX])
        else:
            x_data = np.einsum("ijk->jki", x[i - window_size : i, ...])
            y_data = np.einsum("ij->ji", x[i : i + predict_ahead, :, DISEASE_FEATURE_INDEX])
        features.append(x_data)
        targets.append(y_data)

    dataset = StaticGraphTemporalSignal(
        features=features, targets=targets,
        edge_index=edge_index.t().contiguous(), edge_weight=None,
    )
    train, test = temporal_signal_split(dataset, train_split + valid_split)
    train, valid = temporal_signal_split(train, (valid_split / (train_split + valid_split)))
    return train, valid, test, dataset, mean, std


def create_dataset_single(batch_size, window_size, predict_ahead, self_loop=True,
                           device="cpu", train=0.7, val=0.1,
                           use_disease_only=True, subset=1.0):
    # Returns PyG DataLoaders (for STGAT / DCRNN / ASTGCN / AAGCN).
    adj = load_adjacency_matrix(ADJ_PATH, self_loop)
    edge_index = torch.tensor(
        [[x, y] for x in range(25) for y in range(25) if adj[x][y]], dtype=torch.long
    )
    x, mean, std = load_data(DATA_FILE, use_disease_only, subset)
    dataset = []

    for i in range(window_size, x.shape[0] - predict_ahead):
        if use_disease_only:
            nx = torch.swapaxes(x[i - window_size : i, :, DISEASE_FEATURE_INDEX], 0, 1)
            ny = torch.swapaxes(x[i : i + predict_ahead, :, DISEASE_FEATURE_INDEX], 0, 1)
        else:
            nx = torch.swapaxes(x[i - window_size : i, ...], 0, 1)
            nx = torch.swapaxes(nx, 1, 2)
            ny = torch.swapaxes(x[i : i + predict_ahead, :, DISEASE_FEATURE_INDEX], 0, 1)
        data = Data(x=nx, y=ny, edge_index=edge_index.t().contiguous())
        data.validate(raise_on_error=True)
        data.to(device)
        dataset.append(data)

    drop_last = (len(dataset) % batch_size) != 0
    t_idx = int(train * len(dataset))
    v_idx = int((train + val) * len(dataset))
    train_dl = DataLoader(dataset[:t_idx], batch_size=batch_size, drop_last=drop_last)
    val_dl = DataLoader(dataset[t_idx:v_idx], batch_size=batch_size, drop_last=drop_last)
    test_dl = DataLoader(dataset[v_idx:], batch_size=batch_size, drop_last=drop_last)
    full_dl = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return train_dl, val_dl, test_dl, full_dl, mean, std
"""))

C.append(md(r"""
## 6. Train / infer loops

One pair per model family, since STGAT/DCRNN, A3TGCN, ASTGCN, and AAGCN each expect a
differently-shaped batch. Transcribed from `evaluation.py`, with results collected into
a DataFrame instead of only printed.
"""))

C.append(code(r"""
from torch import optim
from tqdm.auto import tqdm

GNN_RESULTS = []          # rows: {model, segment, split, mae, rmse}
GNN_PREDICTIONS = {}      # model -> (y_pred, y_truth) on the "Full" data, last segment only


@torch.no_grad()
def infer(model, device, dataloader, mean, std, cat):
    model.eval(); model.to(device)
    mae = rmse = mape = n = 0
    y_pred = y_truth = None
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            continue
        pred = model(batch)
        truth = batch.y.view(pred.shape)
        if i == 0:
            y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
        truth = inverse_z_norm(truth, mean, std)
        pred = inverse_z_norm(pred, mean, std)
        y_pred[i, : pred.shape[0], :] = pred
        y_truth[i, : pred.shape[0], :] = truth
        rmse += RMSE(truth, pred); mae += MAE(truth, pred); mape += MAPE(truth, pred); n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"  {cat}: MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}")
    return y_pred, y_truth, float(mae), float(rmse)


def train_model(model, train, val, device, mean, std, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss()
    model.to(device); model.train()
    for epoch in range(epochs):
        for batch in tqdm(train, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            batch = batch.to(device)
            y_pred = model(batch)
            loss = loss_fn(torch.squeeze(y_pred).float(), torch.squeeze(batch.y).float())
            loss.backward(); optimizer.step()
        if not epoch % max(1, epochs // 3):
            print(f"[epoch {epoch+1}] loss={loss:.3f}")
    return model


@torch.no_grad()
def infer_single_shot(model, device, signals, mean, std, cat):
    model.eval(); model.to(device)
    mae = rmse = mape = n = 0
    preds, labels = [], []
    for signal in signals:
        signal = signal.to(device)
        if signal.x.shape[0] == 1:
            continue
        if len(signal.x.shape) == 2:
            pred = torch.squeeze(model(torch.unsqueeze(signal.x, dim=1), signal.edge_index))
        else:
            pred = model(signal.x, signal.edge_index)
        truth = signal.y.view(pred.shape)
        truth = inverse_z_norm(truth, mean, std)
        pred = inverse_z_norm(pred, mean, std)
        preds.append(pred); labels.append(truth)
        rmse += RMSE(truth, pred); mae += MAE(truth, pred); mape += MAPE(truth, pred); n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"  {cat}: MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}")
    return torch.stack(preds), torch.stack(labels), float(mae), float(rmse)


def train_single_shot(model, train, val, device, mean, std, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss()
    model.to(device); model.train()
    for epoch in range(epochs):
        for signal in tqdm(train, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            signal = signal.to(device)
            if len(signal.x.shape) == 2:
                y_pred = model(torch.unsqueeze(signal.x, dim=1), signal.edge_index)
            else:
                y_pred = model(signal.x, signal.edge_index)
            loss = loss_fn(torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float())
            loss.backward(); optimizer.step()
        if not epoch % max(1, epochs // 3):
            print(f"[epoch {epoch+1}] loss={loss:.3f}")
    return model
"""))

C.append(code(r"""
@torch.no_grad()
def infer_ASTGCN(model, device, dataloader, mean, std, cat, batch_size):
    model.eval(); model.to(device)
    mae = rmse = mape = n = 0
    y_pred = y_truth = None
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            continue
        x = torch.reshape(batch.x, (batch_size, num_nodes, -1, window_size))
        pred = torch.squeeze(model(x.float(), batch.edge_index))
        truth = batch.y.view(pred.shape)
        if i == 0:
            y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
        truth = inverse_z_norm(truth, mean, std)
        pred = inverse_z_norm(pred, mean, std)
        y_pred[i, : pred.shape[0], :] = pred
        y_truth[i, : pred.shape[0], :] = truth
        rmse += RMSE(truth, pred); mae += MAE(truth, pred); mape += MAPE(truth, pred); n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"  {cat}: MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}")
    return y_pred, y_truth, float(mae), float(rmse)


def train_ASTGCN(model, train, val, device, mean, std, epochs, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss()
    model.to(device); model.train()
    for epoch in range(epochs):
        for signal in tqdm(train, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            signal = signal.to(device)
            x = torch.reshape(signal.x, (batch_size, num_nodes, -1, window_size))
            y_pred = model(x.float(), signal.edge_index)
            loss = loss_fn(torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float())
            loss.backward(); optimizer.step()
        if not epoch % max(1, epochs // 3):
            print(f"[epoch {epoch+1}] loss={loss:.3f}")
    return model


@torch.no_grad()
def infer_AAGCN(model, device, dataloader, mean, std, cat, batch_size):
    model.eval(); model.to(device)
    mae = rmse = mape = n = 0
    y_pred = y_truth = None
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            continue
        x = torch.reshape(batch.x, (batch_size, -1, window_size, num_nodes))
        pred = torch.reshape(model(x.float()), (batch_size * num_nodes, -1))
        pred = torch.squeeze(pred)
        truth = batch.y.view(pred.shape)
        if i == 0:
            y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
        truth = inverse_z_norm(truth, mean, std)
        pred = inverse_z_norm(pred, mean, std)
        y_pred[i, : pred.shape[0], :] = pred
        y_truth[i, : pred.shape[0], :] = truth
        rmse += RMSE(truth, pred); mae += MAE(truth, pred); mape += MAPE(truth, pred); n += 1
    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"  {cat}: MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}")
    return y_pred, y_truth, float(mae), float(rmse)


def train_AAGCN(model, train, val, device, mean, std, epochs, batch_size):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss()
    model.to(device); model.train()
    for epoch in range(epochs):
        for signal in tqdm(train, desc=f"epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            signal = signal.to(device)
            x = torch.reshape(signal.x, (batch_size, -1, window_size, num_nodes))
            y_pred = torch.reshape(model(x.float()), (batch_size * num_nodes, -1))
            loss = loss_fn(torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float())
            loss.backward(); optimizer.step()
        if not epoch % max(1, epochs // 3):
            print(f"[epoch {epoch+1}] loss={loss:.3f}")
    return model
"""))

C.append(md(r"""## 7. Run each model across the rolling segments"""))

C.append(code(r"""
def run_stgat(segments):
    for i, s in enumerate(segments, 1):
        bs = 2
        train, val, test, full, m, sd = create_dataset_single(
            bs, in_channels, out_channels, use_disease_only=True, subset=s, device=DEVICE
        )
        model = STGAT(in_channels=in_channels, out_channels=out_channels,
                      n_nodes=num_nodes, batch_size=bs, dropout=dropout)
        model = train_model(model, train, val, DEVICE, m, sd, num_epochs)
        _, _, mae_t, rmse_t = infer(model, DEVICE, test, m, sd, "Test")
        yp, yt, mae_f, rmse_f = infer(model, DEVICE, full, m, sd, "Full")
        GNN_RESULTS.append({"model": "STGAT", "segment": s, "split": "test", "mae": mae_t, "rmse": rmse_t})
        GNN_RESULTS.append({"model": "STGAT", "segment": s, "split": "full", "mae": mae_f, "rmse": rmse_f})
        if i == len(segments):
            GNN_PREDICTIONS["STGAT"] = (yp, yt, bs)


def run_a3tgcn(segments):
    for i, s in enumerate(segments, 1):
        use_disease_only = DISEASE_ONLY  # False only once covariates exist (processed mode)
        train, val, test, full, m, sd = create_dataset_single_shot(use_disease_only=use_disease_only, subset=s)
        model = TemporalGCN(1 if use_disease_only else N_FEATURES, 3)
        model = train_single_shot(model, train, val, DEVICE, m, sd, num_epochs)
        _, _, mae_t, rmse_t = infer_single_shot(model, DEVICE, test, m, sd, "Test")
        yp, yt, mae_f, rmse_f = infer_single_shot(model, DEVICE, full, m, sd, "Full")
        GNN_RESULTS.append({"model": "A3TGCN", "segment": s, "split": "test", "mae": mae_t, "rmse": rmse_t})
        GNN_RESULTS.append({"model": "A3TGCN", "segment": s, "split": "full", "mae": mae_f, "rmse": rmse_f})
        if i == len(segments):
            GNN_PREDICTIONS["A3TGCN"] = (yp, yt, 1)


def run_astgcn(segments):
    bs = 1
    for i, s in enumerate(segments, 1):
        use_disease_only = DISEASE_ONLY  # False only once covariates exist (processed mode)
        train, val, test, full, m, sd = create_dataset_single(
            bs, in_channels, out_channels, use_disease_only=use_disease_only, subset=s, device=DEVICE
        )
        model = AttentionSTGCN(num_nodes=25, num_feats=1 if use_disease_only else N_FEATURES,
                                window_size=3, predict_ahead=3).get_model()
        model = train_ASTGCN(model, train, val, DEVICE, m, sd, num_epochs, bs)
        _, _, mae_t, rmse_t = infer_ASTGCN(model, DEVICE, test, m, sd, "Test", bs)
        yp, yt, mae_f, rmse_f = infer_ASTGCN(model, DEVICE, full, m, sd, "Full", bs)
        GNN_RESULTS.append({"model": "ASTGCN", "segment": s, "split": "test", "mae": mae_t, "rmse": rmse_t})
        GNN_RESULTS.append({"model": "ASTGCN", "segment": s, "split": "full", "mae": mae_f, "rmse": rmse_f})
        if i == len(segments):
            GNN_PREDICTIONS["ASTGCN"] = (yp, yt, bs)


def run_dcrnn(segments):
    bs = 1
    for i, s in enumerate(segments, 1):
        use_disease_only = True
        train, val, test, full, m, sd = create_dataset_single(
            bs, in_channels, out_channels, use_disease_only=use_disease_only, subset=s, device=DEVICE
        )
        model = DConvRNN(node_features=window_size * N_FEATURES if not use_disease_only else window_size, num_classes=3)
        model = train_model(model, train, val, DEVICE, m, sd, num_epochs)
        _, _, mae_t, rmse_t = infer(model, DEVICE, test, m, sd, "Test")
        yp, yt, mae_f, rmse_f = infer(model, DEVICE, full, m, sd, "Full")
        GNN_RESULTS.append({"model": "DCRNN", "segment": s, "split": "test", "mae": mae_t, "rmse": rmse_t})
        GNN_RESULTS.append({"model": "DCRNN", "segment": s, "split": "full", "mae": mae_f, "rmse": rmse_f})
        if i == len(segments):
            GNN_PREDICTIONS["DCRNN"] = (yp, yt, bs)


def run_aagcn(segments):
    bs = 1
    for i, s in enumerate(segments, 1):
        use_disease_only = True
        train, val, test, full, m, sd = create_dataset_single(
            bs, in_channels, out_channels, use_disease_only=use_disease_only, subset=s, device=DEVICE
        )
        adj = load_adjacency_matrix(ADJ_PATH, True)
        edge_index = (
            torch.tensor([[x, y] for x in range(25) for y in range(25) if adj[x][y]], dtype=torch.long)
            .t().contiguous().long()
        )
        model = AdaptiveGCN(1 if use_disease_only else N_FEATURES, 1, num_nodes, edge_index).get_model()
        model = train_AAGCN(model, train, val, DEVICE, m, sd, num_epochs, bs)
        _, _, mae_t, rmse_t = infer_AAGCN(model, DEVICE, test, m, sd, "Test", bs)
        yp, yt, mae_f, rmse_f = infer_AAGCN(model, DEVICE, full, m, sd, "Full", bs)
        GNN_RESULTS.append({"model": "AAGCN", "segment": s, "split": "test", "mae": mae_t, "rmse": rmse_t})
        GNN_RESULTS.append({"model": "AAGCN", "segment": s, "split": "full", "mae": mae_f, "rmse": rmse_f})
        if i == len(segments):
            GNN_PREDICTIONS["AAGCN"] = (yp, yt, bs)
"""))

C.append(code(r"""
print("Running STGAT..."); run_stgat(SEGMENTS)
"""))

C.append(code(r"""
print("Running A3TGCN..."); run_a3tgcn(SEGMENTS)
"""))

C.append(code(r"""
print("Running ASTGCN..."); run_astgcn(SEGMENTS)
"""))

C.append(code(r"""
print("Running DCRNN..."); run_dcrnn(SEGMENTS)
"""))

C.append(code(r"""
print("Running AAGCN..."); run_aagcn(SEGMENTS)
"""))

C.append(md(r"""## 8. Aggregate results & compare to the paper"""))

C.append(code(r"""
import pandas as pd, os

os.makedirs("./results", exist_ok=True)
gnn_df = pd.DataFrame(GNN_RESULTS)
gnn_df.to_csv("./results/baseline_gnn_all.csv", index=False)

gnn_leaderboard = (
    gnn_df[gnn_df["split"] == "test"]
    .groupby("model")[["mae", "rmse"]].agg(["mean", "std"]).round(2)
    .sort_values(("rmse", "mean"))
)
gnn_leaderboard
"""))

C.append(code(r"""
print(f"{'Model':<10} {'Our MAE':>10} {'Paper MAE':>10} | {'Our RMSE':>10} {'Paper RMSE':>10}")
for model in gnn_leaderboard.index:
    our_mae = gnn_leaderboard.loc[model, ("mae", "mean")]
    our_rmse = gnn_leaderboard.loc[model, ("rmse", "mean")]
    ref = REFERENCE_CV_SHIFTED.get(model, {})
    print(f"{model:<10} {our_mae:>10.2f} {ref.get('MAE', float('nan')):>10.2f} | "
          f"{our_rmse:>10.2f} {ref.get('RMSE', float('nan')):>10.2f}")

print("\nIf QUICK_TEST=True these will be well off the paper's numbers (few epochs, one "
      "segment) — that's expected. Re-run with QUICK_TEST=False for a real comparison.")
if DISEASE_ONLY:
    print("(DISEASE_ONLY=True — the reference numbers were fit WITH meteorological "
          "covariates, ours without, on a different date range. Treat as directional: "
          "STGAT should still end up lowest-error among these five, matching Weng et al.'s "
          "finding, even if the absolute MAE/RMSE values don't match.)")
"""))

C.append(md(r"""## 9. Forecast plots (paper's Fig. 4 / Fig. 6 style)"""))

C.append(code(r"""
import datetime

def plot_gnn_forecast(model_name, district_idx=None, district_name=None):
    yp, yt, bs = GNN_PREDICTIONS[model_name]
    # shape: (num_batches, bs*num_nodes, predict_ahead) -> (num_batches, bs, num_nodes, predict_ahead)
    yp_r = yp.reshape(yp.shape[0], bs, num_nodes, yp.shape[-1])[:, :, :, 0]
    yt_r = yt.reshape(yt.shape[0], bs, num_nodes, yt.shape[-1])[:, :, :, 0]

    if district_name is not None:
        district_idx = [k for k, v in index.items() if v == district_name][0]
    elif district_idx is None:
        district_idx = 9

    series_pred = yp_r[:, :, district_idx].flatten().detach().cpu().numpy()
    series_true = yt_r[:, :, district_idx].flatten().detach().cpu().numpy()

    start_date = datetime.date(2013, 5, 13)
    t = [start_date + datetime.timedelta(weeks=w) for w in range(len(series_true))]
    split = int(0.7 * len(t))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t[:split], series_true[:split], color=INK_MUTED, linewidth=1.2, label="Train (actual)")
    ax.plot(t[split:], series_true[split:], color=COLORS["blue"], linewidth=1.6, label="Test (actual)")
    ax.plot(t[split:], series_pred[split:], color=COLORS["orange"], linewidth=1.6,
            linestyle="--", label=f"{model_name} (predicted)")
    style_ax(ax)
    ax.set_title(f"{model_name} — {index[district_idx]}", loc="left", fontsize=13)
    ax.set_ylabel("Dengue cases")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


best_gnn = gnn_leaderboard.index[0]
if best_gnn in GNN_PREDICTIONS:
    plot_gnn_forecast(best_gnn, district_name="Colombo" if "Colombo" in index.values() else None)
"""))

C.append(md(r"""
## 10. Combined leaderboard: classical vs. GNN

Loads `01_baselines_classical.ipynb`'s saved results (run that notebook first) and
merges with this notebook's GNN results — this combined table/chart is the headline
result for the project proposal's "Baseline Reproduction" section: it should show the
same qualitative gap Weng et al. report (GNNs beating traditional EWS).
"""))

C.append(code(r"""
classical_path = "./results/baseline_classical_all.csv"
if os.path.exists(classical_path):
    classical_df = pd.read_csv(classical_path)
    classical_summary = classical_df.groupby("model")[["mae", "rmse"]].mean().reset_index()
    classical_summary["family"] = "Classical baseline"

    gnn_summary = (
        gnn_df[gnn_df["split"] == "test"].groupby("model")[["mae", "rmse"]].mean().reset_index()
    )
    gnn_summary["family"] = "Spatio-temporal GNN"

    combined = pd.concat([classical_summary, gnn_summary], ignore_index=True).sort_values("rmse")
    combined.to_csv("./results/combined_leaderboard.csv", index=False)
    display(combined)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(combined))
    bar_colors = [COLORS["blue"] if f == "Classical baseline" else COLORS["orange"] for f in combined["family"]]
    ax.barh(y_pos, combined["rmse"], color=bar_colors, height=0.6)
    ax.set_yticks(y_pos); ax.set_yticklabels(combined["model"])
    ax.invert_yaxis()
    style_ax(ax)
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("Baseline reproduction — classical vs. spatio-temporal GNN", loc="left", fontsize=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COLORS["blue"], label="Classical baseline"),
        Patch(color=COLORS["orange"], label="Spatio-temporal GNN"),
    ], frameon=False, loc="lower right")
    plt.tight_layout()
    plt.show()
else:
    print(f"'{classical_path}' not found — run 01_baselines_classical.ipynb first, "
          "then re-run this cell.")
"""))

C.append(md(r"""
## Next steps — beyond the baseline

This notebook + `01_baselines_classical.ipynb` complete the **baseline reproduction**
milestone: the same models, same protocol, same dataset as Weng et al. (2024).

Per the literature review's recommendations, the next stages are:

1. **Stage 1 — physics-informed loss.** Add an SEIR-SEI (host-vector) residual term to
   the best-performing GNN here (likely STGAT or a Graph-WaveNet-style learned adjacency
   variant): `L = L_data + λ_phys·L_residual + λ_cons·L_conservation + λ_smooth·L_neighbor`.
   Compare 3-step-ahead RMSE and peak-season accuracy against this notebook's GNN-only
   numbers.
2. **Stage 2 — GAN augmentation.** TimeGAN or a conditional GAN (RCGAN-style,
   conditioned on meteorological covariates), with a jittering/window-warping baseline
   for comparison, evaluated by whether it *measurably* improves downstream RMSE/CRPS —
   not just distributional similarity.
3. **Ablation table.** (a) GNN alone [this notebook], (b) GNN + physics loss,
   (c) GNN + GAN augmentation, (d) all three — this is what substantiates the novelty
   claim in the proposal.

See `PROJECT_PROPOSAL_GUIDE.md` (repo root) for how these results map onto the actual
proposal document.
"""))
