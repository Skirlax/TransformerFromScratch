import random

import torch as th

from TransformerModel.Layers.decoder_layer import DecoderLayer
from TransformerModel.Layers.encoder_layer import EncoderLayer
from Embeddings.positional_embeddings import PositionalEmbeddings
# from Embeddings.token_embeddings import decode
from Embeddings.token_embeddings import decode, encode
from tqdm import tqdm
from itertools import chain
import json
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt
from itertools import chain
import wandb


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
                 output_size: int, num_layers: int, vocab_size: int, block_size: int):
        super(TransformerDecoder, self).__init__()
        self.block_size = block_size
        self.embedding = th.nn.Embedding(vocab_size, embedding_dim)
        # self.positional_embedding = PositionalEmbeddings()
        self.positional_embedding = th.nn.Embedding(self.block_size, embedding_dim)
        self.decoder_layers = th.nn.ModuleList(
            [DecoderLayer(embedding_dim, num_heads, dropout_p, output_size) for _ in range(num_layers)])
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.layer_norm = th.nn.LayerNorm([embedding_dim])
        self.lin = th.nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, y):
        # print(self.embedding.weight.shape)
        # print(self.embedding.weight[])
        # x = x.view(-1,self.embedding_dim)
        x1 = self.embedding.forward(x)
        ar = th.arange(0, x.shape[1]).long()
        # print(th.max(ar))
        # print(th.min(ar))
        x2 = self.positional_embedding(ar.to(x.device))
        x = x1 + x2
        for dec_layer in self.decoder_layers:
            x = dec_layer.forward(x, y)
        x = self.layer_norm.forward(x)
        x = self.lin.forward(x)
        return x

    def generate(self, inpt: th.tensor, max_len: int, block_size: int) -> list[list[int]]:
        outputs = []
        for _ in tqdm(range(max_len), desc="Generating text"):
            inpt = inpt[:, -block_size:]
            output = self.forward(inpt, None)
            output = output[:, -1, :]  # get last token
            output = th.softmax(output, dim=-1)
            output = th.multinomial(output, num_samples=1)
            # output = th.argmax(output, dim=-1).unsqueeze(1)
            inpt = th.cat((inpt, output), dim=1)
            outputs.append(output.tolist()[0])
            # print(decode(output.tolist(),vocab))
        return outputs


class TransformerTrainer:
    def __init__(self, transformer: th.nn.Module, optimizer: th.optim,
                 loss_fn: th.nn, lr: float, block_size: int, weight_decay: float, vocab: dict, table):
        self.transformer = transformer
        self.optimizer = optimizer(self.transformer.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss = loss_fn()
        self.lr = lr
        self.table = table
        self.vocab = vocab
        self.block_size = block_size
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.scheduler = None

    def train(self, train, test, epochs: int, batch_size: int, eval_every: int = 500):
        scheduler_epochs = int(epochs * (2 / 3))
        # self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=4e-3, total_steps=scheduler_epochs,
        #                                                   steps_per_epoch=scheduler_epochs // 2)
        # self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,min_lr=1e-8,verbose=True)
        self.transformer.to(self.device)
        self.transformer.train()
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            x_batch, y_batch = self.make_batch(train, batch_size, self.block_size)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.transformer.forward(x_batch, None)
            loss = self.loss(output.view(-1, output.size(-1)), y_batch.view(-1))
            self.train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            if epoch < scheduler_epochs and self.scheduler is not None:
                self.scheduler.step()
            pbar.set_description(f'Epoch: {epoch} Loss: {loss}')
            if epoch % eval_every == 0:
                self.test(test, batch_size, 50)
                acc = self.calculate_accuracy(test, batch_size)
                self.transformer.train()
                wandb.log({"train_loss": loss.item(), "test_loss": self.test_losses[-1], "accuracy": acc,
                           "table_key": self.table})
            else:
                wandb.log({"train_loss": loss.item()})

    @th.no_grad()
    def test(self, test, batch_size: int, test_epochs: int):
        # self.transformer.to(self.device)
        self.transformer.eval()
        pbar = tqdm(range(test_epochs), position=1)
        for epoch in pbar:
            x_batch, y_batch = self.make_batch(test, batch_size, self.block_size)
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            output = self.transformer.forward(x_batch, None)
            otp = th.softmax(output, dim=-1)
            otp = th.argmax(otp, dim=-1)
            otp = otp.tolist()
            decoded = decode(otp, self.vocab)
            # print(len(decoded))
            # print(decoded[63])
            random_choice = random.randint(0, len(decoded) - 1)
            ybt = y_batch.tolist()
            decoded2 = decode(ybt, self.vocab)
            if self.table is not None:
                self.table.add_data(decoded[random_choice:random_choice + 5], decoded2[random_choice:random_choice + 5])
            loss = self.loss(output.view(-1, output.size(-1)), y_batch.view(-1))
            self.test_losses.append(loss.item())
            pbar.set_description(f'Test Epoch: {epoch} Loss: {loss}')

    @th.no_grad()
    def calculate_accuracy(self, test, batch_size: int):
        self.transformer.eval()
        x_batch, y_batch = self.make_batch(test, batch_size, self.block_size)
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        # y_batch = y_batch.unsqueeze(-1)
        output = self.transformer.forward(x_batch, None)
        output = th.softmax(output, dim=-1)
        output = th.argmax(output, dim=-1)
        return accuracy(output, y_batch, task="multiclass", num_classes=len(self.vocab))

        # arg_max = output.argmax(dim=-1, keepdim=True)
        # self.show_most_error_tokens(arg_max, y_batch)
        # equal = (y_batch.unsqueeze(-1) == arg_max).long()
        # sm = th.sum(equal)
        # accuracy = sm / th.numel(y_batch)
        # print(f"Accuracy: {accuracy}")
        # self.accuracies.append(accuracy)

    def show_most_error_tokens(self, predictions: th.tensor, correct: th.tensor):
        not_equal = (predictions != correct.unsqueeze(-1)).long()
        # choose top 3 most error sequences
        not_equal = th.sum(not_equal.squeeze(-1), dim=1)
        top_3 = th.topk(not_equal, 3)
        preds = predictions.tolist()
        preds = list(chain.from_iterable(chain.from_iterable(preds)))
        corr = correct.tolist()
        corr = list(chain.from_iterable(corr))

        # decode
        predictions = decode(preds, self.vocab)
        # matching correct
        correct = decode(corr, self.vocab)
        for index in top_3.indices:
            fig, axs = plt.subplots()
            axs.text(0.05, 0.3, f"Predicted: {predictions}")
            axs.text(0.05, 0.8, f"Correct: {correct}")
            plt.savefig(f"error_{index}.png")

    def make_batch(self, data, batch_size: int, block_size: int):
        data = data.view(-1)
        random_start = th.randint(0, len(data) - block_size, (batch_size,))
        x_batch = th.stack([data[i:i + block_size] for i in random_start])
        # y is shifted by one
        y_batch = th.stack([data[i + 1:i + block_size + 1] for i in random_start])
        return x_batch, y_batch

    def batch_data(self, data, batch_size, block_size, data_size):
        data_copy = data.clone()
        data_copy = data_copy.view(-1)
        for x in range(data_size // batch_size):  # how many iterations in one epoch
            random_start = th.randint(0, len(data_copy) - block_size, (batch_size,))
            x_batch = th.stack([data_copy[i:i + block_size] for i in random_start])
            y_batch = th.stack([data_copy[i + 1:i + block_size + 1] for i in random_start])
            data_copy_ind = th.arange(0, data_copy.size(0), 1)
            data_copy = data_copy[~th.isin(data_copy_ind, random_start)]
            yield x_batch, y_batch

    def draw_losses(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.test_losses, label="Test Loss")
        plt.legend()
        plt.show()
