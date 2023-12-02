import re
from utils import project_root


def prepare_to_tokenize() -> list:
    with open(f"{project_root()}/Datasets/TinyShakespeare/input.txt", "r") as f:
        text = f.read()
    text = text.lower()
    text = text.replace("\n\n", "\n")
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    text = text.strip()
    text_list = text.split("\n")
    text_list = [i for i in text_list if i != ""]
    return text_list

# if __name__ == "__main__":
#     prepare_to_tokenize()