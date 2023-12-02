import json

from Datasets.TinyShakespeare.prepare import prepare_to_tokenize
from tqdm import tqdm
from Embeddings.constants import EmbedConstants


class BytePairDecoder:
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.corpus_chars = self.corpus_characters()
        self.vocab = self.make_vocab_from_freqs(self.get_freqs())

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

    def corpus_characters(self) -> list:
        chars = []
        for word in self.make_words():
            chars.extend(list(word) + ["</wb>"])  # wb = word boundary

        return chars

    def get_freqs(self):
        freqs = {}
        boundary_indexes = [i for i, x in enumerate(self.corpus_chars) if x == "</wb>"]
        boundary_indexes.insert(0, -1)
        for i in range(len(boundary_indexes) - 1):
            b_idx = boundary_indexes[i]
            next_b_idx = boundary_indexes[i + 1]
            word_chars = ",".join(self.corpus_chars[b_idx + 1:next_b_idx])
            freqs.setdefault(word_chars, 0)
            freqs[word_chars] += 1
        return freqs

    def byte_pair_encoding(self, target_vocab_size):
        for i in tqdm(range(target_vocab_size - len(self.vocab)), desc="Byte Pair Encoding"):
            # print(f"Current vocab size: {len(self.vocab)}")
            freqs = self.get_freqs()
            freqs_keys_list = list(freqs.keys())
            conv_pairs = ["".join(x.split(",")[i:i + 2]) for x in freqs_keys_list for i in
                          range(len(x.split(",")) - 1)]
            conv_pairs_with_freqs = [["".join(x.split(",")[i:i + 2])] * freqs[x] for x in freqs_keys_list for i in
                                     range(len(x.split(",")) - 1)]
            pair_freqs = {}
            for i in range(len(conv_pairs)):
                pair_freqs.setdefault(conv_pairs[i], 0)
                pair_freqs[conv_pairs[i]] += conv_pairs_with_freqs[i].count(conv_pairs[i])

            if len(pair_freqs) == 0:
                break

            max_freq_pair = max(pair_freqs,
                                key=pair_freqs.get)
            self.vocab.append("".join(max_freq_pair))
            self.merge_to_corpus(max_freq_pair)

    def merge_to_corpus(self, merge_pair):
        for index, char in enumerate(self.corpus_chars):
            if index == len(self.corpus_chars) - 1:
                break
            next_char = self.corpus_chars[index + 1]
            if "".join([char, next_char]) == "".join(merge_pair):
                self.corpus_chars[index] = merge_pair
                self.corpus_chars.pop(index + 1)

    def get_vocab(self):
        return self.vocab

    def prepare_vocab(self):
        self.vocab.remove("")
        self.vocab.append(" ")
        self.vocab.append(EmbedConstants.PAD)
        self.vocab.append(EmbedConstants.SOS)
        self.vocab.append(EmbedConstants.EOS)
        self.vocab.append(EmbedConstants.UNK)
        for index, char in enumerate(self.vocab):
            if char.isnumeric():
                self.vocab.append(self.vocab.pop(index))

    def get_indexed_vocab(self):
        return {x: i for i, x in enumerate(self.vocab)}


def encode(inputs: list[str], vocab: dict, pad: bool = True, eos: bool = True, sos: bool = True) -> list[list[int]]:
    for index, sentence in enumerate(inputs):
        for key in reversed(vocab.keys()):
            inputs[index] = inputs[index].replace(key, f"|{vocab[key]}|")

    inputs = list(map(lambda x: [int(y) for y in x.split("|") if y.isnumeric()], inputs))
    max_len = max(inputs, key=len)
    for index, sentence in enumerate(inputs):
        if pad:
            inputs[index] = sentence + [vocab[EmbedConstants.PAD]] * (len(max_len) - len(sentence))
        else:
            inputs[index] = sentence
        if sos:
            inputs[index] = [vocab[EmbedConstants.SOS]] + inputs[index]
        if eos:
            inputs[index] = inputs[index] + [vocab[EmbedConstants.EOS]]
    return inputs


def decode(inputs: list[list[int]], vocab: dict, skip_special: bool = True):
    i_vocab = {v: k for k, v in vocab.items()}
    res = list(map(lambda x: "".join(
        [i_vocab[y] for y in x if (i_vocab[y] not in EmbedConstants.SPECIAL_TOKENS if skip_special else True)]),
                   inputs))
    return res

# if __name__ == "__main__":
#     encoder = BytePairDecoder(prepare_to_tokenize())
#     encoder.byte_pair_encoding(500)
#     encoder.move_numbers_to_end()
#     with open("vocab.json", "w") as f:
#         json.dump(encoder.get_indexed_vocab(), f)
