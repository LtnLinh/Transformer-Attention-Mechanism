import torch
import torch.nn as nn
from torch.profiler import record_function


class TransformerBase(nn.Module):
    """One transformer block: embedding -> attention -> FFN.

    Embedding, projections, layer norms and FFN always run in PyTorch on the
    CPU; subclasses only swap out the attention step (the profiled bottleneck).
    """

    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, d_ff=128 * 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def attention(self, q, k, v):
        """Scaled dot-product attention: [B, N, D] x3 -> [B, N, D]."""
        raise NotImplementedError()

    def forward(self, x):
        # x: [B, N] token ids
        with record_function("1_embedding"):
            h = self.embedding(x)  # [B, N, D]

        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)

        with record_function("2_attention_total"):
            attn_out = self.attention(q, k, v)

        h = self.norm1(h + attn_out)

        with record_function("3_ffn"):
            h = self.norm2(h + self.ffn(h))

        return h
