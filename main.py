import torch as th
from Datasets.TinyShakespeare.prepare import prepare_to_tokenize
# from Embeddings.token_embeddings import encode, decode
from Embeddings.token_embeddings import gpt2_encode, gpt2_decode
from TransformerModel.transformer import TransformerDecoder, TransformerTrainer
import json
import numpy as np
from itertools import chain
from Embeddings.constants import EmbedConstants
import random
import optuna
import pickle


def main():
    # text = prepare_to_tokenize(
    #     "/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon.txt")
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/gpt2_vocab-vocab.json",
    #           "r") as file:
    #     vocab = json.load(file)
    # text = prepare_to_tokenize("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/TinyShakespeare/input.txt")
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Embeddings/tiny_s_final.json", "r") as file:
    #     vocab = json.load(file)
    # text = encode(text, vocab)
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/encoded_big.pkl",
              "rb") as file:
        text = pickle.load(file)

    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/vocab_big.json", "r") as file:
        vocab = json.load(file)
    # text = gpt2_encode(open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon_encoded.pkl","r").)
    embedd_dim = 450
    num_heads = 6
    dropout_p = 0.55
    feed_forward_size = 916
    num_layers = 12
    vocab_size = len(vocab)
    # vocab_size = 50257
    # test_lr = 1e-5
    test_lr = 3e-4
    block_size = 108
    weight_decay = 0.01
    # weight_decay = 0.007548162081615722
    epochs = 10_000
    transformer = TransformerDecoder(embedd_dim, num_heads, dropout_p, feed_forward_size, num_layers, vocab_size,
                                     block_size)
    trainer = TransformerTrainer(transformer, th.optim.AdamW, th.nn.CrossEntropyLoss, test_lr, block_size, weight_decay,
                                 vocab, None)
    data = list(chain.from_iterable(text))
    # take 10% for test,10% for validation and 80% for training

    test = data[:int(len(data) * 0.1)]
    val = data[int(len(data) * 0.1):int(len(data) * 0.2)]
    train = data[int(len(data) * 0.2):]
    train = th.tensor(train).long()
    # val = th.tensor(val).long()
    test = th.tensor(test).long()
    batch_size = 8

    # print("cuda" if th.cuda.is_available() else "cpu")
    # print(ts_data > 503)
    trainer.train(train, test, epochs, batch_size)
    th.save({
        "model_state_dict": transformer.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
    }, "model.pth")
    trainer.draw_losses()

    input_ = gpt2_encode(["Eragon"])
    input_ = th.tensor(input_).long().cuda().reshape(1, -1)
    outputs = gpt2_decode(transformer.generate(input_, 1000, block_size))
    print(outputs)


def gen():
    # text = prepare_to_tokenize("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/TinyShakespeare/input.txt")
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Embeddings/tiny_s_final.json", "r") as file:
    #     vocab = json.load(file)

    # text = prepare_to_tokenize("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/HungerGames/m1.2.txt")
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Embeddings/hg_final.json", "r") as file:
    #     vocab = json.load(file)
    # text = prepare_to_tokenize(
    #     "/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/eragon_cz.txt")
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/vocab_cz.json", "r") as file:
        vocab = json.load(file)

    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon_encoded.pkl",
              "r") as file:
        text = pickle.load(file)["input_ids"]

    embedd_dim = 450
    num_heads = 7
    dropout_p = 0.2011408235618257
    feed_forward_size = 916
    num_layers = 12
    vocab_size = len(vocab)
    test_lr = 3e-4
    block_size = 108
    weight_decay = 0.007548162081615722
    epochs = 10_000
    transformer = TransformerDecoder(embedd_dim, num_heads, dropout_p, feed_forward_size, num_layers, vocab_size,
                                     block_size)
    data_ = th.load("model.pth")
    transformer.load_state_dict(data_["model_state_dict"])
    transformer.to("cuda")
    transformer.eval()
    inpt = ""
    while inpt != "q":
        inpt = input("Enter a sentence: ")
        input_ = gpt2_encode([inpt])
        input_ = th.tensor(input_).long().cuda().reshape(1, -1)
        gen_ = transformer.generate(input_, 64 * 4, block_size)
        # gen_ = chain.from_iterable(gen_)
        outp = gpt2_decode(gen_)
        print("".join(outp))
    # print(outputs)

    # trainer.train(x, y, 15, 5)


def param_search(trial_epochs: int, num_trials: int):
    text = prepare_to_tokenize(
        "/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/inheritance_text.txt")
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/edited_vocab.json", "r") as file:
        vocab = json.load(file)
    text = gpt2_encode(text)

    def objective(trial):
        embedd_dim = trial.suggest_int("embedd_dim", 120, 540, step=6)
        num_heads = trial.suggest_int("num_heads", 4, 12)
        dropout_p = trial.suggest_float("dropout_p", 0.1, 0.6)
        feed_forward_size = trial.suggest_int("feed_forward_size", 600, 1800)
        num_layers = trial.suggest_int("num_layers", 4, 12)
        block_size = trial.suggest_int("block_size", 64, 128)
        lr = trial.suggest_float("lr", 1e-6, 7e-3)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1)
        batch_size = 32
        transformer = TransformerDecoder(embedd_dim, num_heads, dropout_p, feed_forward_size, num_layers, len(vocab),
                                         block_size)
        trainer = TransformerTrainer(transformer, th.optim.AdamW, th.nn.CrossEntropyLoss, lr, block_size, weight_decay)
        data = list(chain.from_iterable(text))
        test = data[:int(len(data) * 0.1)]
        # val = data[int(len(data) * 0.1):int(len(data) * 0.2)]
        train = data[int(len(data) * 0.2):]
        train = th.tensor(train).long()
        test = th.tensor(test).long()
        batch_size = find_batch_size(batch_size, trainer, train, test)
        del transformer
        del trainer
        transformer = TransformerDecoder(embedd_dim, num_heads, dropout_p, feed_forward_size, num_layers, len(vocab),
                                         block_size)
        trainer = TransformerTrainer(transformer, th.optim.AdamW, th.nn.CrossEntropyLoss, lr, block_size, weight_decay)
        trainer.train(train, test, trial_epochs, batch_size)
        return np.mean(trainer.test_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")
    with open("params.json", "w") as file:
        json.dump(study.best_params, file)


def find_batch_size(batch_size: int, trainer: TransformerTrainer, train, test):
    while True:
        try:
            trainer.train(train, test, 1, batch_size)
            break
        # except oom
        except RuntimeError:
            batch_size //= 2
            print("Batch size too big, reducing to ", batch_size)
    return batch_size


if __name__ == "__main__":
    # vocab = json.load(open("vocab2.json", "r"))
    # print(encode(["all "],vocab,eos=False,sos=False,pad=False))
    main()
    # param_search(501, 100)
    # gen()
