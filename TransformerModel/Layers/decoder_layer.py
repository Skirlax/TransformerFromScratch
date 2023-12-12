import torch as th

from TransformerModel.Blocks.feed_forward import FeedForward
from TransformerModel.Blocks.mutli_head_attention import MultiHeadAttention


class DecoderLayer(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float, output_size: int):
        super(DecoderLayer, self).__init__()
        self.self_masked_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm1 = th.nn.LayerNorm([embedding_dim])
        self.cross_attn = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm2 = th.nn.LayerNorm([embedding_dim])
        self.feed_forward = FeedForward(embedding_dim, output_size)
        self.layer_norm3 = th.nn.LayerNorm([embedding_dim])
        self.dropout = th.nn.Dropout(p=dropout_p)

    def forward(self, x, encoder_output):
        x = self.layer_norm1.forward(x)
        self_masked_attn_output = self.self_masked_attn.forward(x.clone(), mask=True)
        x += self.dropout(self_masked_attn_output)
        x = self.layer_norm2.forward(x)
        cross_attn_out = self.cross_attn.forward(x.clone(), encoder_output=encoder_output,mask=True)
        x += self.dropout(cross_attn_out)
        x = self.layer_norm3.forward(x)
        feed_forward_out = self.feed_forward.forward(x.clone())
        x += self.dropout(feed_forward_out.clone())
        return x
