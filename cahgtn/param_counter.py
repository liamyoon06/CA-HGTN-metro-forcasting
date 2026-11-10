"""
param_counter.py -- analytical, standard-PyTorch parameter counts for
CA-HGTN and the baselines, computed directly from documented parameterization
formulas rather than via the project's numpy-based test shim.

WHY ANALYTICAL, NOT VIA THE SHIM: the shim's TransformerEncoderLayer and GRU
are deliberately simplified stand-ins for testing FORWARD-PASS shapes and
logic cheaply -- they do not reproduce real PyTorch's actual internal
parameter structure (e.g. the shim's "TransformerEncoderLayer" is a single
Linear layer, nowhere close to real multi-head attention + two feedforward
layers + two LayerNorms). Using the shim's `parameters()` here would give a
confidently-wrong, badly undercounted number. This file instead uses the
standard, documented PyTorch formulas directly, cross-checked below against
hand-computable cases.

Formulas used (standard, single-source-of-truth for nn.Module parameter
counts in PyTorch, as documented in the PyTorch source/docs):
  Linear(in, out, bias=True):        in*out + out
  LayerNorm(d) [elementwise_affine]: 2*d
  GRU(input, hidden, bidirectional, num_layers):
      per layer, per direction: 3*hidden*(layer_input + hidden + 2)
      (layer_input = `input` for layer 0, else hidden*num_directions)
  TransformerEncoderLayer(d_model, dim_feedforward):
      self_attn (in_proj + out_proj): 4*d_model^2 + 4*d_model
      linear1 + linear2:              2*d_model*dim_feedforward + d_model + dim_feedforward
      norm1 + norm2:                  4*d_model
      (nn.TransformerEncoder(layer, num_layers=L) deep-copies the layer L
      times -- L INDEPENDENT sets of parameters, not shared/tied weights)
"""


def linear_params(in_f: int, out_f: int, bias: bool = True) -> int:
    return in_f * out_f + (out_f if bias else 0)


def layernorm_params(d: int) -> int:
    return 2 * d


def gru_params(input_size: int, hidden_size: int, bidirectional: bool = False, num_layers: int = 1) -> int:
    num_directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else hidden_size * num_directions
        per_direction = 3 * hidden_size * (layer_input + hidden_size + 2)
        total += per_direction * num_directions
    return total


def transformer_encoder_layer_params(d_model: int, dim_feedforward: int) -> int:
    self_attn = 4 * d_model * d_model + 4 * d_model
    feedforward = 2 * d_model * dim_feedforward + d_model + dim_feedforward
    norms = 4 * d_model
    return self_attn + feedforward + norms


# ----------------------------------------------------------------
# Unit tests against hand-computable cases -- run before trusting
# the totals below on the real architecture.
# ----------------------------------------------------------------
def _run_self_tests():
    # Linear(10, 5): 10*5 + 5 = 55
    assert linear_params(10, 5) == 55, linear_params(10, 5)
    # Linear(4, 4, bias=False): 16
    assert linear_params(4, 4, bias=False) == 16

    # LayerNorm(64): 2*64 = 128
    assert layernorm_params(64) == 128

    # GRU(input=8, hidden=4, single layer, unidirectional):
    # 3*4*(8+4+2) = 3*4*14 = 168
    assert gru_params(8, 4) == 168, gru_params(8, 4)
    # Same but bidirectional: double (two independent direction-sets of weights)
    assert gru_params(8, 4, bidirectional=True) == 336

    # TransformerEncoderLayer(d_model=8, dim_feedforward=32):
    # self_attn = 4*64 + 32 = 256+32 = 288
    # feedforward = 2*8*32 + 8 + 32 = 512+40 = 552
    # norms = 32
    # total = 288+552+32 = 872
    got = transformer_encoder_layer_params(8, 32)
    assert got == 872, got

    print("param_counter self-tests: ALL PASSED")


_run_self_tests()


def multigraph_attention_params(hidden_dim: int, num_stations: int, graph_mode: str = "full") -> int:
    n_graphs = 3 if graph_mode == "full" else 1
    node_emb = 2 * num_stations * 16          # node_emb1 + node_emb2, each (num_stations, 16)
    qkv = 3 * linear_params(hidden_dim, hidden_dim)
    graph_gate = n_graphs                       # nn.Parameter(torch.ones(n_graphs))
    out_proj = linear_params(hidden_dim, hidden_dim)
    norm = layernorm_params(hidden_dim)
    return node_emb + qkv + graph_gate + out_proj + norm


def temporal_fusion_params(hidden_dim: int, num_heads: int = 4, num_layers: int = 2) -> int:
    per_layer = transformer_encoder_layer_params(hidden_dim, hidden_dim * 4)
    transformer = per_layer * num_layers  # nn.TransformerEncoder deep-copies -- independent params per layer
    gru = gru_params(hidden_dim, hidden_dim, bidirectional=True)
    gru_proj = linear_params(hidden_dim * 2, hidden_dim)
    gate = linear_params(hidden_dim * 2, hidden_dim)
    return transformer + gru + gru_proj + gate


def cahgtn_params(hidden_dim: int, num_stations: int, horizon: int = 4,
                   spatial_layers: int = 2, head_type: str = "nb", graph_mode: str = "full") -> int:
    input_proj = linear_params(2, hidden_dim)
    time_emb = linear_params(4, hidden_dim)
    extra_proj = linear_params(4, hidden_dim)
    spatial = spatial_layers * multigraph_attention_params(hidden_dim, num_stations, graph_mode)
    temporal = temporal_fusion_params(hidden_dim)
    trunk = linear_params(hidden_dim, hidden_dim)  # nn.Sequential(Linear, ReLU, Dropout) -- only Linear has params
    if head_type == "nb":
        heads = linear_params(hidden_dim, horizon * 2) * 2  # mu_head + r_head
    else:
        heads = linear_params(hidden_dim, horizon * 2)       # point_head only
    return input_proj + time_emb + extra_proj + spatial + temporal + trunk + heads


def gru_only_baseline_params(hidden_dim: int, horizon: int = 4) -> int:
    input_proj = linear_params(6, hidden_dim)
    gru = gru_params(hidden_dim, hidden_dim, bidirectional=False)
    head = linear_params(hidden_dim, horizon * 2)
    return input_proj + gru + head


def gcn_gru_baseline_params(hidden_dim: int, horizon: int = 4, bidirectional_attention: bool = False,
                             temporal_mode: str | None = None) -> int:
    if temporal_mode is None:
        temporal_mode = "bidirectional_attention" if bidirectional_attention else "plain"
    gcn = linear_params(2, hidden_dim)          # GCNLayer's single Linear(2, hidden_dim)
    time_proj = linear_params(4, hidden_dim)
    if temporal_mode == "stacked_bidir_uni_attention":
        gru_bidir = gru_params(hidden_dim, hidden_dim, bidirectional=True)
        gru_uni = gru_params(hidden_dim * 2, hidden_dim, bidirectional=False)
        attn_pool = linear_params(hidden_dim, 1)
        head = linear_params(hidden_dim, horizon * 2)
        return gcn + time_proj + gru_bidir + gru_uni + attn_pool + head
    elif temporal_mode == "bidirectional_attention":
        gru = gru_params(hidden_dim, hidden_dim, bidirectional=True)
        attn_pool = linear_params(hidden_dim * 2, 1)
        head = linear_params(hidden_dim * 2, horizon * 2)
        return gcn + time_proj + gru + attn_pool + head
    else:
        gru = gru_params(hidden_dim, hidden_dim, bidirectional=False)
        head = linear_params(hidden_dim, horizon * 2)
        return gcn + time_proj + gru + head


if __name__ == "__main__":
    for city, N in [("Hangzhou", 80), ("Shanghai", 288)]:
        print(f"\n=== {city} (N={N} stations) ===")
        for hd in [64, 128, 256, 512]:
            c = cahgtn_params(hd, N, head_type="nb", graph_mode="full")
            g = gru_only_baseline_params(hd)
            b = gcn_gru_baseline_params(hd)
            print(f"  hidden_dim={hd:4d}: CA-HGTN(full)={c:>10,}   GRU-only={g:>8,}   GCN+GRU={b:>8,}")
