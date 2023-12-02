import re


def prepare_to_tokenize() -> tuple[list, list]:
    with open("Datasets/multi30K/train_en", "r",encoding="utf-8") as file:
        x = file.readlines()
    with open("Datasets/multi30K/train_cs", "r",encoding="utf-8") as file:
        y = file.readlines()

    pattern = r'[^a-zA-Z0-9\s]'
    x = [re.sub(pattern, '', i) for i in x]
    y = [re.sub(pattern, '', i) for i in y]
    return x, y
