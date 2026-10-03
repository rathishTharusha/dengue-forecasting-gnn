# %% [markdown]
# ## 4. The six encoders, written out
#
# The five published spatio-temporal GNNs (Weng et al. 2024) and the LSTM of the
# SEIR-LSTM (Liu et al. 2025), in plain PyTorch. Nothing is imported from a graph
# library: with 25 districts every graph operation is a small dense matrix
# product, so each layer is written as the equation it implements.
#
# Each one is a **transcription** of the implementation the benchmark used
# (`torch_geometric_temporal` 0.54 and PyTorch Geometric's `GATConv`), wired the
# way the benchmark wires it. Parameter names match those implementations, so a
# weight-for-weight comparison is possible; the repository test
# `tests/test_kaggle_architectures.py` loads the reference weights into every
# module below and checks the outputs agree to floating-point precision.
#
# Two properties of the reference implementations are kept on purpose, because
# they are part of the published models' behaviour:
#
# * **A3TGCN** restarts its GRU state at zero for every input week and
#   attention-weights the per-week outputs; it is not a recurrence over the window.
# * **DCRNN**'s diffusion convolution pairs its reverse-direction weights with the
#   edge list *by position*, and its Chebyshev-style recursion resets the older
#   term to the input at every step. Both are reproduced exactly.
#
# Each encoder maps a window of fold-scaled log cases `(batch, 25, 3)` to a
# representation `(batch, 25, 64)` that the forecast heads (section 5) read.

# %%
import math

import torch
import torch.nn.functional as F
from torch import nn


def glorot_(t: torch.Tensor) -> torch.Tensor:
    """PyG's glorot: uniform in +/- sqrt(6 / (fan_in + fan_out)) on the last two dims."""
    stdv = math.sqrt(6.0 / (t.size(-2) + t.size(-1)))
    with torch.no_grad():
        return t.uniform_(-stdv, stdv)


def dense_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """``A[src, dst] = 1`` for every edge; duplicates are not expected."""
    a = torch.zeros(n, n)
    a[edge_index[0], edge_index[1]] = 1.0
    return a


class Head(nn.Module):
    """The reference's single output layer (``improved.make_head`` with no increments).

    ``nodes == 0``: applied to every node, ``(..., in) -> (..., out)``.
    ``nodes > 0`` : emits every node from one graph-level vector (STGAT), ``(B, in) -> (B, nodes, out)``.
    """

    def __init__(self, in_features: int, out: int, nodes: int = 0):
        super().__init__()
        self.out, self.nodes = out, nodes
        self.linear = nn.Linear(in_features, max(nodes, 1) * out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.linear(h)
        return y.reshape(*h.shape[:-1], self.nodes, self.out) if self.nodes else y


# %% [markdown]
# ### STGAT — graph attention over districts, then two LSTMs over time
#
# A graph-attention layer (8 heads, averaged) mixes each week's case values
# across neighbouring districts:
#
# $\alpha_{ij} = \mathrm{softmax}_{j \in \mathcal{N}(i) \cup \{i\}}\big(\mathrm{LeakyReLU}(a_s^\top W x_j + a_d^\top W x_i)\big)$,
# $\;x_i' = \tfrac{1}{8}\sum_{\text{heads}} \sum_j \alpha_{ij} W x_j + b$.
#
# The attended window then enters two LSTMs **with the 25 districts as the feature
# axis**, so the LSTM output is one graph-level vector per sample, and a linear layer
# emits every district from it — the benchmark's wiring.

# %%
class GATConv(nn.Module):
    """PyG ``GATConv(in, out, heads, concat=False, dropout)`` on a dense 25-node graph."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super().__init__()
        self.heads, self.out_channels = heads, out_channels
        self.dropout, self.negative_slope = dropout, negative_slope
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        glorot_(self.lin.weight)
        glorot_(self.att_src)
        glorot_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        """``x`` (B, N, in); ``incoming[i, j] = 1`` if j sends to i (self-loops included)."""
        b, n, _ = x.shape
        xp = self.lin(x).view(b, n, self.heads, self.out_channels)            # (B,N,H,C)
        a_src = (xp * self.att_src).sum(-1)                                   # (B,N,H)
        a_dst = (xp * self.att_dst).sum(-1)
        e = F.leaky_relu(a_dst[:, :, None, :] + a_src[:, None, :, :], self.negative_slope)  # (B,i,j,H)
        e = e.masked_fill(incoming[None, :, :, None] == 0, float("-inf"))
        alpha = F.dropout(torch.softmax(e, dim=2), p=self.dropout, training=self.training)
        out = torch.einsum("bijh,bjhc->bihc", alpha, xp)
        return out.mean(dim=2) + self.bias


class STGAT(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor,
                 heads: int = 8, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.gat = GATConv(window, window, heads=heads, dropout=dropout)
        self.lstm1 = nn.LSTM(n_nodes, hidden, num_layers=1)
        self.lstm2 = nn.LSTM(hidden, hidden, num_layers=1)
        self.head = Head(hidden, out, nodes=n_nodes)
        for lstm in (self.lstm1, self.lstm2):
            for name, p in lstm.named_parameters():
                (nn.init.constant_(p, 0.0) if "bias" in name else nn.init.xavier_uniform_(p))
        nn.init.xavier_uniform_(self.head.linear.weight)
        a = dense_adjacency(edge_index, n_nodes)
        incoming = a.T.clone()                     # incoming[i, j] = A[j, i]
        incoming.fill_diagonal_(1.0)               # GATConv removes then re-adds self-loops
        self.register_buffer("incoming", incoming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        h = F.dropout(self.gat(x, self.incoming), self.dropout, training=self.training)
        h = h.movedim(2, 0)                        # (window, B, N): districts are the LSTM features
        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        return self.head(h[-1])


# %% [markdown]
# ### A3TGCN — a graph-convolutional GRU per week, attention over weeks
#
# The GCN is $\tilde{A} X W + b$ with $\tilde{A} = D^{-1/2} A^\top D^{-1/2}$ over the
# graph with self-loops and $D$ the in-degree. The TGCN cell is a GRU whose gates
# are GCNs of the input concatenated with the state. A3TGCN runs that cell on each
# of the 3 input weeks from a zero state and combines the results with
# softmax-normalised learned weights over weeks.

# %%
class GCNConv(nn.Module):
    """PyG ``GCNConv`` (symmetric normalisation, self-loops) on a dense graph."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        glorot_(self.lin.weight)

    @staticmethod
    def propagation(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        idx = torch.arange(a.shape[0])
        a[idx, idx] = torch.where(a[idx, idx] > 0, a[idx, idx], torch.ones_like(a[idx, idx]))  # remaining self-loops
        d = a.sum(0).pow(-0.5)                                # in-degree (PyG normalises on the target)
        return (d[:, None] * a * d[None, :]).T                # out[dst] = sum_src M[dst, src] x[src]

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return m @ self.lin(x) + self.bias


class TGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv_z = GCNConv(in_channels, out_channels)
        self.linear_z = nn.Linear(2 * out_channels, out_channels)
        self.conv_r = GCNConv(in_channels, out_channels)
        self.linear_r = nn.Linear(2 * out_channels, out_channels)
        self.conv_h = GCNConv(in_channels, out_channels)
        self.linear_h = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, m: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(*x.shape[:-1], self.out_channels)
        z = torch.sigmoid(self.linear_z(torch.cat([self.conv_z(x, m), h], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([self.conv_r(x, m), h], -1)))
        h_tilde = torch.tanh(self.linear_h(torch.cat([self.conv_h(x, m), h * r], -1)))
        return z * h + (1 - z) * h_tilde


class A3TGCNCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, periods: int):
        super().__init__()
        self.periods = periods
        self._base_tgcn = TGCN(in_channels, out_channels)
        self._attention = nn.Parameter(torch.empty(periods))
        nn.init.uniform_(self._attention)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``x`` (B, N, in_channels, periods)."""
        probs = torch.softmax(self._attention, dim=0)
        return sum(probs[p] * self._base_tgcn(x[..., p], m, None) for p in range(self.periods))


class A3TGCN(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor, hidden: int = 32):
        super().__init__()
        self.core = A3TGCNCore(1, hidden, window)
        self.head = Head(hidden, out)
        self.register_buffer("m", GCNConv.propagation(dense_adjacency(edge_index, n_nodes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.core(x.unsqueeze(2), self.m)))


# %% [markdown]
# ### ASTGCN — spatial and temporal attention around a Chebyshev graph convolution
#
# Two blocks. Each computes temporal attention $E$ (a softmax over week pairs) and
# spatial attention $S$ (a softmax over district pairs), then a K = 3 Chebyshev
# convolution on the scaled Laplacian $\tilde{L} = 2L/\lambda_{\max} - I$ with
# $L = D_{\text{out}} - A$, where the first-order term is modulated elementwise by
# $S$, followed by a (1×3) convolution over weeks, a residual connection and layer
# norm. A final convolution over weeks produces the 16-channel representation.

# %%
class ChebConvAttention(nn.Module):
    """``torch_geometric_temporal`` ``ChebConvAttention`` with ``normalization=None``, dense."""

    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self._weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        self._bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self._weight)
        nn.init.uniform_(self._bias)

    @staticmethod
    def scaled_laplacian(a: torch.Tensor) -> torch.Tensor:
        import numpy as np
        a = a.clone()
        a.fill_diagonal_(0.0)                                   # remove_self_loops
        lap = torch.diag(a.sum(1)) - a                          # get_laplacian(None): D_out - A
        ev = np.linalg.eigvals(lap.double().numpy())            # LaplacianLambdaMax: largest |eig|
        lam = float(ev[np.argmax(np.abs(ev))].real)
        return 2.0 * lap / lam - torch.eye(a.shape[0])

    def forward(self, x: torch.Tensor, lt: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        tx0 = torch.matmul((torch.eye(n) * s).permute(0, 2, 1), x)
        out = torch.matmul(tx0, self._weight[0])
        if self._weight.size(0) > 1:
            tx1 = torch.matmul(lt * s, tx0)                     # attention on the first order only
            out = out + torch.matmul(tx1, self._weight[1])
        for k in range(2, self._weight.size(0)):
            tx2 = 2.0 * torch.matmul(lt, tx1) - tx0
            out = out + torch.matmul(tx2, self._weight[k])
            tx0, tx1 = tx1, tx2
        return out + self._bias


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lhs = torch.matmul(torch.matmul(x, self._W1), self._W2)
        rhs = torch.matmul(self._W3, x).transpose(-1, -2)
        s = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(lhs, rhs) + self._bs))
        return F.softmax(s, dim=1)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self._U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self._U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self._U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self._be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self._Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self._U1), self._U2)
        rhs = torch.matmul(self._U3, x)
        e = torch.matmul(self._Ve, torch.sigmoid(torch.matmul(lhs, rhs) + self._be))
        return F.softmax(e, dim=1)


class ASTGCNBlock(nn.Module):
    def __init__(self, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int,
                 time_strides: int, num_of_vertices: int, num_of_timesteps: int):
        super().__init__()
        self._temporal_attention = TemporalAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._spatial_attention = SpatialAttention(in_channels, num_of_vertices, num_of_timesteps)
        self._chebconv_attention = ChebConvAttention(in_channels, nb_chev_filter, K)
        self._time_convolution = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3),
                                           stride=(1, time_strides), padding=(0, 1))
        self._residual_convolution = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1),
                                               stride=(1, time_strides))
        self._layer_norm = nn.LayerNorm(nb_time_filter)
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x: torch.Tensor, lt: torch.Tensor) -> torch.Tensor:
        b, n, f, t = x.shape
        x_tilde = torch.matmul(x.reshape(b, -1, t), self._temporal_attention(x)).reshape(b, n, f, t)
        s = self._spatial_attention(x_tilde)
        x_hat = F.relu(torch.cat([self._chebconv_attention(x[:, :, :, k], lt, s).unsqueeze(-1)
                                  for k in range(t)], dim=-1))
        x_hat = self._time_convolution(x_hat.permute(0, 2, 1, 3))
        x = self._residual_convolution(x.permute(0, 2, 1, 3))
        x = self._layer_norm(F.relu(x + x_hat).permute(0, 3, 2, 1))
        return x.permute(0, 2, 3, 1)


class ASTGCNCore(nn.Module):
    def __init__(self, nb_block: int, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int,
                 time_strides: int, num_for_predict: int, len_input: int, num_of_vertices: int):
        super().__init__()
        self._blocklist = nn.ModuleList(
            [ASTGCNBlock(in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                         num_of_vertices, len_input)]
            + [ASTGCNBlock(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, num_of_vertices,
                           len_input // time_strides) for _ in range(nb_block - 1)])
        self._final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict,
                                     kernel_size=(1, nb_time_filter))
        for p in self.parameters():
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x: torch.Tensor, lt: torch.Tensor) -> torch.Tensor:
        for block in self._blocklist:
            x = block(x, lt)
        x = self._final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1]
        return x.permute(0, 2, 1)


class ASTGCN(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor, feat: int = 16):
        super().__init__()
        # The reference sets num_for_predict = horizon, making the last convolution the
        # forecast; widening it to `feat` and projecting gives the same seam as the others.
        self.core = ASTGCNCore(nb_block=2, in_channels=1, K=3, nb_chev_filter=64, nb_time_filter=64,
                               time_strides=1, num_for_predict=feat, len_input=window,
                               num_of_vertices=n_nodes)
        self.head = Head(feat, out)
        self.register_buffer("lt", ChebConvAttention.scaled_laplacian(dense_adjacency(edge_index, n_nodes)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.core(x.unsqueeze(2), self.lt))


# %% [markdown]
# ### DCRNN — a diffusion-convolutional GRU
#
# The diffusion convolution sums $K = 32$ powers of the forward and reverse random-walk
# transition matrices applied to the input, each with its own weights. The GRU gates
# are diffusion convolutions of the input concatenated with the state. The benchmark
# feeds the 3-week window as input channels and takes one GRU step from a zero state.

# %%
class DConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    @staticmethod
    def transitions(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense forms of the reference's two message-passing operators.

        Forward: each edge (s, d) carries x[s] / deg_out[s] into d.
        Reverse: the reference builds the reversed edge list from the transposed
        dense adjacency (row-major order) but indexes its weights with the ORIGINAL
        list's source nodes, position by position. That pairing is reproduced as is.
        """
        a = dense_adjacency(edge_index, n)
        deg_out, deg_in = a.sum(1), a.sum(0)
        src, dst = torch.nonzero(a, as_tuple=True)              # row-major, as the reference's list
        p_out = torch.zeros(n, n)
        p_out[dst, src] = 1.0 / deg_out[src]
        rsrc, rdst = torch.nonzero(a.T, as_tuple=True)          # dense_to_sparse(A^T), row-major
        w_in = 1.0 / deg_in[src]                                # norm_in = deg_in_inv[row], by position
        p_in = torch.zeros(n, n)
        p_in.index_put_((rdst, rsrc), w_in, accumulate=True)
        return p_out, p_in

    def forward(self, x: torch.Tensor, p_out: torch.Tensor, p_in: torch.Tensor) -> torch.Tensor:
        tx0 = tx1 = x
        h = torch.matmul(tx0, self.weight[0][0]) + torch.matmul(tx0, self.weight[1][0])
        if self.weight.size(1) > 1:
            tx1_o, tx1_i = p_out @ x, p_in @ x
            h = h + torch.matmul(tx1_o, self.weight[0][1]) + torch.matmul(tx1_i, self.weight[1][1])
        for k in range(2, self.weight.size(1)):
            tx2_o = 2.0 * (p_out @ tx1_o) - tx0
            tx2_i = 2.0 * (p_in @ tx1_i) - tx0
            h = h + torch.matmul(tx2_o, self.weight[0][k]) + torch.matmul(tx2_i, self.weight[1][k])
            tx0, tx1_o, tx1_i = tx1, tx2_o, tx2_i                # tx1 is never updated: tx0 stays x
        return h + self.bias


class DCRNNCell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv_x_z = DConv(in_channels + out_channels, out_channels, K)
        self.conv_x_r = DConv(in_channels + out_channels, out_channels, K)
        self.conv_x_h = DConv(in_channels + out_channels, out_channels, K)

    def forward(self, x, p_out, p_in, h=None):
        if h is None:
            h = torch.zeros(*x.shape[:-1], self.out_channels)
        z = torch.sigmoid(self.conv_x_z(torch.cat([x, h], -1), p_out, p_in))
        r = torch.sigmoid(self.conv_x_r(torch.cat([x, h], -1), p_out, p_in))
        h_tilde = torch.tanh(self.conv_x_h(torch.cat([x, h * r], -1), p_out, p_in))
        return z * h + (1 - z) * h_tilde


class DCRNN(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor,
                 hidden: int = 64, K: int = 32):
        super().__init__()
        self.core = DCRNNCell(window, hidden, K)
        self.head = Head(hidden, out)
        p_out, p_in = DConv.transitions(edge_index, n_nodes)
        self.register_buffer("p_out", p_out)
        self.register_buffer("p_in", p_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.core(x, self.p_out, self.p_in)))


# %% [markdown]
# ### AAGCN — attention-augmented graph convolution with a temporal convolution
#
# Three graph partitions (self, inward, outward — column-normalised adjacency and its
# transpose), a 1×1 convolution per partition, batch norm and a residual; then
# spatial, temporal and channel attention; then a (9×1) temporal convolution. The
# benchmark runs it non-adaptive (the learned-adjacency branch off) with 8 channels.

# %%
class UnitTCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")
        nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, coff_embedding: int = 4,
                 num_subset: int = 3):
        super().__init__()
        self.out_c, self.num_subset = out_channels, num_subset
        self.register_buffer("A", A)
        num_jpts = A.shape[-1]
        self.conv_d = nn.ModuleList(nn.Conv2d(in_channels, out_channels, 1) for _ in range(num_subset))
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=(ker_jpt - 1) // 2)
        self.fc1c = nn.Linear(out_channels, out_channels // 2)
        self.fc2c = nn.Linear(out_channels // 2, out_channels)
        self.down = (nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
                     if in_channels != out_channels else nn.Identity())
        self.bn = nn.BatchNorm2d(out_channels)
        # Initialisation exactly as the reference, in its order.
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.bn.weight, 1e-6)
        nn.init.constant_(self.bn.bias, 0)
        for conv in self.conv_d:
            n, k1, k2 = conv.weight.size(0), conv.weight.size(1), conv.weight.size(2)
            nn.init.normal_(conv.weight, 0, math.sqrt(2.0 / (n * k1 * k2 * num_subset)))
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        n, c, t, v = x.size()
        y = None
        for i in range(self.num_subset):
            z = self.conv_d[i](torch.matmul(x.reshape(n, c * t, v), self.A[i]).view(n, c, t, v))
            y = z if y is None else z + y
        y = F.relu(self.bn(y) + self.down(x))
        se = torch.sigmoid(self.conv_sa(y.mean(-2)))                           # spatial attention
        y = y * se.unsqueeze(-2) + y
        se = torch.sigmoid(self.conv_ta(y.mean(-1)))                           # temporal attention
        y = y * se.unsqueeze(-1) + y
        se = torch.sigmoid(self.fc2c(F.relu(self.fc1c(y.mean(-1).mean(-1)))))  # channel attention
        return y * se.unsqueeze(-1).unsqueeze(-1) + y


class AAGCNCore(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.gcn1 = UnitGCN(in_channels, out_channels, A)
        self.tcn1 = UnitTCN(out_channels, out_channels)
        self.residual = UnitTCN(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class AAGCN(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor, channels: int = 8):
        super().__init__()
        a = dense_adjacency(edge_index, n_nodes)
        inward = F.normalize(a, dim=0, p=1)
        outward = F.normalize(a.T, dim=0, p=1)
        self.core = AAGCNCore(1, channels, torch.stack([torch.eye(n_nodes), inward, outward]))
        self.head = Head(channels * window, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.core(x.permute(0, 2, 1).unsqueeze(1))                         # (B, C, T, N)
        b, c, t, n = o.shape
        return self.head(o.permute(0, 3, 1, 2).reshape(b, n, c * t))


# %% [markdown]
# ### LSTM — the SEIR-LSTM's encoder, no graph at all
#
# Two LSTM layers over each district's own 3-week window. It is the control that says
# what the graph adds.

# %%
class LSTMEncoder(nn.Module):
    def __init__(self, n_nodes: int, window: int, out: int, edge_index: torch.Tensor | None = None,
                 layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, out, num_layers=layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        h, _ = self.lstm(x.reshape(b * n, w, 1))
        return h[:, -1].reshape(b, n, -1)


ENCODERS = {"STGAT": STGAT, "A3TGCN": A3TGCN, "ASTGCN": ASTGCN, "DCRNN": DCRNN, "AAGCN": AAGCN,
            "LSTM": LSTMEncoder}


class Encoder(nn.Module):
    """One of the six as the representation for the heads.

    The case window goes through the architecture at its reference contract; any
    exogenous channels (seasonal features, climate) go around it and are joined at
    the head, so the published architecture runs exactly as designed.
    """

    def __init__(self, name: str, n_nodes: int, window: int, extra: int, hidden: int,
                 edge_index: torch.Tensor):
        super().__init__()
        self.window, self.extra = window, extra
        self.core = ENCODERS[name](n_nodes, window, hidden, edge_index)
        self.out_dim = hidden + extra

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.core(x[..., : self.window].contiguous()))
        return torch.cat([h, x[..., self.window:]], dim=-1) if self.extra else h
