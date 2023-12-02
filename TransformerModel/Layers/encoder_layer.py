import torch as th
from TransformerModel.Blocks.mutli_head_attention import MultiHeadAttention
from TransformerModel.Blocks.feed_forward import FeedForward


class EncoderLayer(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm1 = th.nn.LayerNorm([embedding_dim])
        self.feed_forward = FeedForward(embedding_dim, 128)
        self.layer_norm2 = th.nn.LayerNorm([embedding_dim])
        self.dropout = th.nn.Dropout(p=dropout_p)

    def forward(self, x):
        attn_out = self.attention.forward(x)
        # RESHAPE
        x += self.dropout(attn_out)
        x = self.layer_norm1.forward(x)

        forward_out = self.feed_forward.forward(x)
        # RESHAPE
        x += self.dropout(forward_out)
        x = self.layer_norm2.forward(x)

        return x
