import torch as th


class MultiHeadAttention(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.one_head_size = embedding_dim // num_heads

        # UNVECTORIZED:
        # self.Qs = th.nn.ParameterList(
        #     [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
        #      range(num_heads)])
        # self.Ks = th.nn.ParameterList(
        #     [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
        #      range(num_heads)])
        # self.Vs = th.nn.ParameterList(
        #     [th.nn.Parameter(th.nn.init.kaiming_normal_(th.empty(1, self.one_head_size, embedding_dim))) for _ in
        #      range(num_heads)])

        # VECTORIZED:
        self.q = th.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = th.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = th.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.lin = th.nn.Linear(self.one_head_size * self.num_heads, embedding_dim)

    def masked_attention(self, x, mask: bool, encoder_output=None) -> th.tensor:
        outputs = []
        if mask:
            mask = self.make_mask(x)
        for i, (q, k, v) in enumerate(zip(self.Qs, self.Ks, self.Vs)):
            Q = x @ q.transpose(-1, -2)  # T -> (batch_size,embedding_dim,one_head_size)
            if encoder_output is None:
                K = x @ k.transpose(-1, -2)  # T -> (batch_size,embedding_dim,one_head_size)
                V = x @ v.transpose(-1, -2)  # T -> (batch_size,embedding_dim,one_head_size)
            else:
                K = encoder_output @ k.transpose(-1, -2)
                V = encoder_output @ v.transpose(-1, -2)
            # tmp = th.matmul(Q, K.transpose(-2, -1))
            raw_scores = th.matmul(Q, K.transpose(-2, -1)) / th.sqrt(th.tensor(
                self.embedding_dim))  # (batch_size,embedding_dim,one_head_size) @ (batch_size,one_head_size,embedding_dim) -> (batch_size,embedding_dim,embedding_dim)
            scores = th.softmax(raw_scores + mask, dim=-1)
            value_scores = scores @ V
            outputs.append(value_scores)

        res = th.cat(outputs, dim=-1)
        res = self.lin(res)

        return res

    def masked_attention_vectorized(self, x, mask: bool, encoder_output=None) -> th.tensor:
        Q = self.split_heads(self.q(x))
        if encoder_output is None:
            K = self.split_heads(self.k(x))
            V = self.split_heads(self.v(x))
        else:
            K = self.split_heads(self.k(encoder_output))
            V = self.split_heads(self.v(encoder_output))

        raw_scores = th.matmul(Q, K.transpose(-2, -1)) / th.sqrt(th.tensor(
            self.embedding_dim))  # (batch_size,num_heads,seq_len,one_head_size) @ (batch_size,num_heads,one_head_size,seq_len) -> (batch_size,num_heads,seq_len,seq_len)
        if mask:
            mask = self.make_mask(x)
            raw_scores += mask
        scores = th.softmax(raw_scores, dim=-1)  # -1 is sequence length
        value_scores = th.matmul(scores, V)
        outputs = self.combine_heads(value_scores)
        return self.lin(outputs)

    def make_mask(self, x: th.Tensor):
        size = (self.num_heads, x.size(1), x.size(1))
        mask = th.tril(th.ones(size=size,device=x.device))
        return mask.masked_fill(mask == 0, float('-inf')).cuda()

    def split_heads(self, mult_output: th.tensor):
        return mult_output.view(mult_output.size(0), -1, self.num_heads, self.one_head_size).transpose(1,
                                                                                                       2)  # output -> (batch_size,num_heads,seq_len,one_head_size) -1 is sequence length

    def combine_heads(self, mult_output: th.tensor):
        orig_split = mult_output.transpose(1, 2).contiguous()
        return orig_split.reshape(mult_output.size(0), -1,
                                  self.embedding_dim)

    def forward(self, x, mask: bool = False, encoder_output: None or th.tensor = None) -> th.tensor:
        # outputs = self.masked_attention(x, mask, encoder_output=encoder_output)
        outputs = self.masked_attention_vectorized(x, mask, encoder_output=encoder_output)
        return outputs
