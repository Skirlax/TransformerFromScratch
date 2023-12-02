import torch as th


class MultiHeadAttention(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.one_head_size = embedding_dim // num_heads
        self.Qs = th.nn.ParameterList(
            [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
             range(num_heads)])
        self.Ks = th.nn.ParameterList(
            [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
             range(num_heads)])
        self.Vs = th.nn.ParameterList(
            [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
             range(num_heads)])

        self.lin = th.nn.Linear(self.one_head_size * self.num_heads, embedding_dim)

    def masked_attention(self, x, mask: bool, encoder_output=None) -> th.tensor:
        outputs = []
        if mask:
            mask = self.make_mask(x)
        for i, (q, k, v) in enumerate(zip(self.Qs, self.Ks, self.Vs)):
            Q = x @ q.transpose(-1, -2)
            if encoder_output is None:
                K = x @ k.transpose(-1, -2)
                V = x @ v.transpose(-1, -2)
            else:
                K = encoder_output @ k.transpose(-1, -2)
                V = encoder_output @ v.transpose(-1, -2)
            tmp = th.matmul(Q, K.transpose(-2, -1))
            raw_scores = th.matmul(Q, K.transpose(-2, -1)) / th.sqrt(th.tensor(self.embedding_dim))
            scores = th.softmax(raw_scores + mask, dim=-1)
            value_scores = scores @ V
            outputs.append(value_scores)

        res = th.cat(outputs, dim=-1)
        res = self.lin(res)

        return res

    def make_mask(self, x: th.tensor):
        size = (x.size(0), x.size(1), x.size(1))
        mask = th.tril(th.ones(size=size))
        return mask.masked_fill(mask == 0, float('-inf')).cuda()

    def forward(self, x, mask: bool = False, encoder_output: None or th.tensor = None) -> th.tensor:
        outputs = self.masked_attention(x, mask, encoder_output=encoder_output)
        return outputs
