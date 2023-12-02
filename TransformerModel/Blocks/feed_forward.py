import torch as th


class FeedForward(th.nn.Module):
    def __init__(self, embedding_dim: int, output_size: int):
        super(FeedForward, self).__init__()
        self.lin1 = th.nn.Linear(embedding_dim, output_size)
        self.lin2 = th.nn.Linear(output_size, embedding_dim)
        self.activation = th.nn.ReLU()

    def forward(self, x):
        x = self.lin1.forward(x)
        x = self.activation.forward(x)
        x = self.lin2.forward(x)
        return x
