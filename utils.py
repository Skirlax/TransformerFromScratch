import os


def project_root() -> str:
    while "README.md" not in os.listdir():
        os.chdir("..")
    return os.getcwd()
