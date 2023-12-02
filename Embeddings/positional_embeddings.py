import torch as th


class PositionalEmbeddings(th.nn.Module):
    def __init__(self):
        super(PositionalEmbeddings, self).__init__()
        self.requires_grad_(False)

    def forward(self, token_embeddings: th.Tensor) -> th.tensor:
        d = token_embeddings.shape[-1]
        # (batch_size,seq_len,embedd_dim)
        const = 10_000
        pos = th.arange(token_embeddings.shape[1]).unsqueeze(1).cuda()
        i = th.arange(d).unsqueeze(0).cuda()
        embeds = th.where(i % 2 == 0, th.sin(pos / (const ** ((2 * i) / d))), th.cos(pos / (const ** ((2 * i) / d))))
        token_embeddings += embeds.cuda()
        return token_embeddings
