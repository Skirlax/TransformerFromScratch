import torch as th

from TransformerModel.Layers.decoder_layer import DecoderLayer
from TransformerModel.Layers.encoder_layer import EncoderLayer
from Embeddings.positional_embeddings import PositionalEmbeddings
from tqdm import tqdm
from itertools import chain

class TransformerEncoderDecoder(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float,
                 output_size: int, num_layers: int, vocab_size: int):
        super(TransformerEncoderDecoder, self).__init__()
        self.embedding = th.nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbeddings()
        self.positional_embedding.requires_grad_(False)
        self.encoder_layers = th.nn.ModuleList(
            [EncoderLayer(embedding_dim, num_heads, dropout_p) for _ in range(num_layers)])
        self.decoder_layers = th.nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, dropout_p, output_size) for _ in range(num_layers)])
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layer_norm = th.nn.LayerNorm([embedding_dim])
        self.lin = th.nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, y):
        for enc_layer in self.encoder_layers:
            x = enc_layer.forward(x)
        for dec_layer in self.decoder_layers:
            y = dec_layer.forward(y, x)
        x = self.layer_norm.forward(x)
        x = self.lin.forward(x)
        return x


class TransformerDecoder(th.nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float,
                 output_size: int, num_layers: int, vocab_size: int):
        super(TransformerDecoder, self).__init__()
        self.embedding = th.nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbeddings()
        self.decoder_layers = th.nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, dropout_p, output_size) for _ in range(num_layers)])
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layer_norm = th.nn.LayerNorm([embedding_dim])
        self.lin = th.nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, y):
        # print(self.embedding.weight.shape)
        # print(self.embedding.weight[])
        x = self.embedding.forward(x)
        x = self.positional_embedding.forward(x)
        for dec_layer in self.decoder_layers:
            x = dec_layer.forward(x, y)
        x = self.layer_norm.forward(x)
        x = self.lin.forward(x)
        return x

    def generate(self, inpt: th.tensor, max_len: int,block_size: int) -> list[list[int]]:
        # outputs = []
        # inpt = inpt.reshape(1,-1).cuda()
        for _ in tqdm(range(max_len), desc="Generating text"):
            inpt = inpt[:,-block_size:]
            output = self.forward(inpt, None)
            output = output[:, -1, :]  # get last token
            output = th.softmax(output, dim=-1)
            output = th.multinomial(output, num_samples=1)
            inpt = th.cat((inpt,output),dim=1)
        return inpt.tolist()


class TransformerTrainer:
    def __init__(self, transformer: th.nn.Module, optimizer: th.optim,
                 loss_fn: th.nn, lr: float,block_size: int):
        self.transformer = transformer
        self.optimizer = optimizer(self.transformer.parameters(), lr=lr)
        self.loss = loss_fn()
        self.lr = lr
        self.block_size = block_size
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

    def train(self, data, epochs: int, batch_size: int):
        self.transformer.to(self.device)
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            x_batch, y_batch = self.make_batch(data, batch_size, self.block_size)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.transformer.forward(x_batch, None)
            loss = self.loss(output.view(-1, output.size(-1)), y_batch.view(-1))
            loss.backward()
            self.optimizer.step()
            pbar.set_description(f'Epoch: {epoch} Loss: {loss}')
            # print(f'Epoch: {epoch} Loss: {loss}')

    def make_batch(self, data, batch_size: int, block_size: int):
        data = data.view(-1)
        random_start = th.randint(0, len(data) - block_size, (batch_size,))
        x_batch = th.stack([data[i:i + block_size] for i in random_start])
        # y is shifted by one
        y_batch = th.stack([data[i + 1:i + block_size + 1] for i in random_start])
        return x_batch, y_batch

