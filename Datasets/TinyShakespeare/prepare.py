import re
from utils import project_root


def prepare_to_tokenize(path: str) -> list:
    with open(path, "r", encoding='utf-8') as f:
        text = f.read()
    # chars = list(text)
    text = text.replace("'", "").replace('"', "")
    text = text.replace("''", "").replace('""', "")
    text = text.replace("(", "").replace(")", "")
    # text = text.encode("utf-8").decode("utf-8")
    text = text.replace("", "")
    text = re.sub(r'\n\n+', '\n\n', text)
    # text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.split("\n\n")
    for index, part in enumerate(text):
        text[index] = part.replace("\n", " ")
    return text

# if __name__ == "__main__":
#     prepare_to_tokenize()
