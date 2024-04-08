import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseSpAggregatedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q):
        batch_size = q.shape[0]
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        # attention_scores=torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        av = (F.silu(attention_scores) @ v)
        return av.transpose(1, 2).flatten(2)


class HSTUBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.f1 = nn.Linear(d_model, d_model * 4)  # Transform and split
        self.pointwise_attn = PointwiseAggregatedAttention(d_model, num_heads)
        self.f2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        #TODO
        self.rab_p_t = None

    def split(self, x):
        U, V, Q, K = x.chunk(4, dim=-1)
        return U, V, Q, K

    def forward(self, x):
        # Pointwise Projection
        x_proj = F.silu(self.f1(x))
        u, v, q, k = self.split(x_proj)

        # Spatial Aggregation
        av = self.pointwise_attn(v,k,q)

        # Pointwise Transformation
        y = self.f2(self.norm(av * u))

        return y

class GenRec(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([HSTUBlock(d_model, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x
        return x


if __name__ == "__main__":
    model = GenRec(d_model=52, num_heads=2, num_layers=3)
    x = torch.rand(32, 10, 52)
