import torch as th
from Datasets.TinyShakespeare.prepare import prepare_to_tokenize
from Embeddings.token_embeddings import encode,decode
from TransformerModel.transformer import TransformerDecoder, TransformerTrainer
import json
import numpy as np
from itertools import chain
from Embeddings.constants import EmbedConstants
def main():
    with open('Datasets/TinyShakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    # chars = sorted(list(set(text)))
    # vocab_size = len(chars)
    # create a mapping from characters to integers
    # stoi = {ch: i for i, ch in enumerate(chars)}
    # itos = {i: ch for i, ch in enumerate(chars)}
    # encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    # decode = lambda l: ''.join([itos[i] for i in l])  #
    data = prepare_to_tokenize()
    vocab = json.load(open("vocab2.json", "r"))
    data = encode(data, vocab, pad=False,eos=False,sos=False)
    # th.autograd.set_detect_anomaly(True)
    # data = encode(text)
    embedd_dim = 300
    num_heads = 6
    dropout_p = 0.1
    output_size = 300
    num_layers = 6
    vocab_size = len(vocab)
    test_lr = 2e-4
    block_size = 256
    transformer = TransformerDecoder(embedd_dim, num_heads, dropout_p, output_size, num_layers, vocab_size)
    trainer = TransformerTrainer(transformer, th.optim.AdamW, th.nn.CrossEntropyLoss, test_lr,block_size)
    data = list(chain.from_iterable(data))
    ts_data = th.tensor(np.array(data))
    print("cuda" if th.cuda.is_available() else "cpu")
    # print(ts_data > 503)
    trainer.train(ts_data,500,128)
    input_ = encode(["all "],vocab,eos=False,sos=False,pad=False)
    # input_ = th.zeros(1,1).long().cuda()
    # input_ = th.zeros((1,1)).long().cuda()
    input_ = th.tensor(input_).long().cuda().reshape(1,-1)
    outputs = decode(transformer.generate(input_,10,block_size),vocab,skip_special=False)
    print(outputs)

    # trainer.train(x, y, 15, 5)


if __name__ == "__main__":
    # vocab = json.load(open("vocab2.json", "r"))
    # print(encode(["all "],vocab,eos=False,sos=False,pad=False))
    main()
