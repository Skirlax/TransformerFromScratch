import json
import re
from collections import Counter
from itertools import chain
from tqdm import tqdm
from Datasets.TinyShakespeare.prepare import prepare_to_tokenize
import copy
from transformers import GPT2Tokenizer
import pickle


class EmbedConstants:
    SOS = "</SOS>"
    EOS = "</EOS>"
    PAD = "</PAD>"
    UNK = "</UNK>"
    SPECIAL_TOKENS = [SOS, EOS, PAD, UNK]


class BytePairDecoder:
    """
    This class represents a byte pair decoder.
    It takes a corpus as input and performs various operations to generate a vocabulary.

    Args:
        corpus (list): A list of sentences in the corpus.

    Attributes:
        corpus (list): A list of sentences in the corpus.
        corpus_chars (list): A list of characters in the corpus, with word boundaries marked.
        vocab (list): The vocabulary generated from the corpus.

    Methods:
        get_unique_words: Returns a list of unique words in the corpus.
        make_vocab_from_freqs: Generates a vocabulary from word frequencies.
        make_words: Splits the corpus into words.
        split_corpus_with_word_boundary: Splits the corpus into characters, with word boundaries marked.
        get_word_frequencies: Returns the frequency of each word in the corpus.
        byte_pair_encoding: Performs byte pair encoding to generate a vocabulary of target size.
        merge_to_corpus: Merges character pairs in the corpus based on given merge pair.
        get_vocab: Returns the vocabulary.
        prepare_vocab: Prepares the vocabulary for indexing by removing empty strings and adding special tokens.
        get_indexed_vocab: Returns an indexed version of the vocabulary.

    """

    def __init__(self, corpus: list):
        self.corpus = corpus
        self.corpus_chars = self.split_corpus_with_word_boundary()
        self.vocab = self.make_vocab_from_freqs(self.get_word_frequencies())

    def get_unique_words(self) -> list:
        return list(set(self.make_words()))

    def make_vocab_from_freqs(self, freqs: dict) -> list:
        vocab = []
        for key in freqs.keys():
            vocab.extend(key.split(","))
        return list(set(vocab))

    def make_words(self) -> list:
        words = []
        for sentence in self.corpus:
            words.extend(sentence.split(" "))
        return words

    def split_corpus_with_word_boundary(self) -> list:
        chars = []
        for word in self.make_words():
            chars.extend(list(word) + ["</wb>"])  # wb = word boundary

        # chars = [x for x in chars if x != "" and x != '']
        return chars

    def get_word_frequencies(self):
        # join all characters in the corpus into a single string
        corpus_string = ",".join(self.corpus_chars)

        # split the string into words using the word boundary tag
        words = corpus_string.split("</wb>")
        words = [word.strip(',') for word in corpus_string.split("</wb>") if re.search('[a-zA-Z]', word) is not None]
        # count the frequency of each word in the list
        word_frequencies = Counter(words)

        return word_frequencies

    def byte_pair_encoding(self, target_vocab_size):
        for i in tqdm(range(target_vocab_size - len(self.vocab)), desc="Byte Pair Encoding"):
            # print(f"Current vocab size: {len(self.vocab)}")
            freqs = self.get_word_frequencies()
            freqs_keys_list = list(freqs.keys())
            # conv_pairs = ["".join(x.split(",")[i:i + 2]) for x in freqs_keys_list for i in
            #               range(len(x.split(",")) - 1) if len("".join(x.split(",")[i:i + 2])) > 0]
            conv_pairs_with_freqs = [[",".join(x.split(",")[i:i + 2])] * freqs[x] for x in freqs_keys_list for i in
                                     range(len(x.split(",")) - 1) if len("".join(x.split(",")[i:i + 2])) > 0]

            # tmp = list(chain.from_iterable(conv_pairs_with_freqs))
            pair_freqs = Counter(chain.from_iterable(conv_pairs_with_freqs))
            # for i in range(len(conv_pairs)):
            #     pair_freqs.setdefault(conv_pairs[i], 0)
            #     pair_freqs[conv_pairs[i]] += conv_pairs_with_freqs[i].count(conv_pairs[i])

            if len(pair_freqs) == 0:
                break

            max_freq_pair = pair_freqs.most_common(1)[0][0]
            self.vocab.append("".join(max_freq_pair.split(",")))
            self.merge_to_corpus(max_freq_pair.split(","))

    def merge_to_corpus(self, merge_pair):

        merged_corpus = []

        char1, char2 = merge_pair
        merge_pair = "".join(merge_pair)

        i = 0
        while i < len(self.corpus_chars) - 1:
            if self.corpus_chars[i] == char1 and self.corpus_chars[i + 1] == char2:
                merged_corpus.append(merge_pair)
                i += 2  # Skip next char as it is part of the merged pair
            else:
                merged_corpus.append(self.corpus_chars[i])
                i += 1

        if i < len(self.corpus_chars):  # If there are any chars left which could not be merged
            merged_corpus.extend(self.corpus_chars[i:])

        self.corpus_chars = merged_corpus

    def get_vocab(self):
        return self.vocab

    def prepare_vocab(self):
        self.vocab.remove("")
        self.vocab.extend(
            [" ", "\n", ".", ",", EmbedConstants.PAD, EmbedConstants.SOS, EmbedConstants.EOS, EmbedConstants.UNK])
        self.vocab = list(sorted(self.vocab, reverse=False, key=lambda token: int(token) if token.isdigit() else -1))

    def get_indexed_vocab(self):
        return {x: i for i, x in enumerate(self.vocab)}


def encode(inputs: list[str], vocab: dict, pad: bool = True, eos: bool = True, sos: bool = True) -> list[list[int]]:
    subwords = [k for k, v in vocab.items()]
    subwords = sorted(subwords, key=len, reverse=True)
    # rev_voc = {v: k for k, v in vocab.items()}
    encoded = []
    for text in inputs:
        sub_encoded = []
        pattern = re.compile(r'(' + '|'.join(map(re.escape, subwords)) + r')')
        splitted = pattern.split(text)
        splitted = list(filter(lambda x: x != "", splitted))
        for subword in splitted:
            if subword in subwords:
                sub_encoded.append(vocab[subword])
            else:
                sub_encoded.append(vocab[EmbedConstants.UNK])

        encoded.append(sub_encoded)
    return encoded


def reorder_index(vocab__: dict):
    vocab__ = {k: idx for idx, k in enumerate(vocab__.keys())}
    return vocab__


def decode(inputs: list[list[int]], vocab: dict, skip_special: bool = True):
    i_vocab = {v: k for k, v in vocab.items()}
    res = list(map(lambda x: "".join(
        [i_vocab[y] for y in x if (i_vocab[y] not in EmbedConstants.SPECIAL_TOKENS if skip_special else True)]),
                   inputs))
    return res


def gpt2_encode(corpus: list) -> list:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.add_special_tokens({"pad_token": EmbedConstants.PAD, "bos_token": EmbedConstants.SOS,
    #                                 "eos_token": EmbedConstants.EOS, "unk_token": EmbedConstants.UNK})
    return tokenizer(corpus)

def gpt2_decode(inputs: list):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer.decode(inputs)





if __name__ == "__main__":
    # res = gpt2_tokenize(prepare_to_tokenize("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon.txt"))
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon_encoded.pkl", "wb") as file:
    #     pickle.dump(res, file)
    # with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon_encoded.pkl", "rb") as file:
    #     res = pickle.load(file)
    #
    # print(gpt2_decode(res["input_ids"][0]))
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # tokenizer.save_vocabulary("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/",filename_prefix="gpt2_vocab")


    encoder = BytePairDecoder(
        prepare_to_tokenize("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/big_eragon.txt"))
    encoder.byte_pair_encoding(1000)
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/vocab_big.json", "w",
              encoding='utf-8') as file:
        json.dump(encoder.get_indexed_vocab(), file, ensure_ascii=False)
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/vocab_big.json", "r",
              encoding='utf-8') as file:
        vocab = json.load(file)

    encoder.vocab = list(vocab.keys())

    encoder.prepare_vocab()
    vocab_ = encoder.get_indexed_vocab()
    vocab_ = reorder_index(vocab_)
    with open("/home/skyr/PycharmProjects/TransformerFromScratch/Datasets/Inheritance/vocab_big.json", "w",
              encoding='utf-8') as vocab_file:
        json.dump(vocab_, vocab_file, ensure_ascii=False)
