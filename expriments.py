import os
import sys
from pathlib import Path

from helper import ProjectPath


def runs(
    model: int,
    bsize: int,
    ds: str,
    message: str,
    log: bool = True,
    dry_run: bool = False,
    skip: bool = False,
    torch_model: bool = False,
    paper_train: bool = True,
    pretrained: bool = False,
):
    interpreter_path = sys.executable
    base_cmd: str = interpreter_path + " " + str(ProjectPath.base / "main.py") + " "
    if log:
        base_cmd += "-l "
    if dry_run:
        base_cmd += "-d "
    if skip:
        base_cmd += "-s "
    if torch_model:
        base_cmd += "-tm "
    if paper_train:
        base_cmd += "-pt "

    base_cmd += f"-md {model}"
    base_cmd += f"-bz {bsize}"
    base_cmd += f"-ds {ds}"
    base_cmd += f"-p {pretrained}"
    base_cmd += f"-m \"{message}\""
    
    os.system(f"{base_cmd}")


if __name__ == "__main__":
    runs(model=34, bsize=128, ds="Cifar100", message="train my resnet34 without skip", torch_model=True, skip=True, paper_train=False)

    # runs(model=34, bsize=128, ds="Cifar100", message="train my resnet34 without skip")
    # runs(model=34, bsize=128, ds="Cifar100", message="train my resnet34 without skip", torch_model=True)

    # runs(model=50, bsize=64, ds="Cifar100", message="train my resnet50 without skip")
    # runs(model=50, bsize=64, ds="Cifar100", message="train my resnet50 without skip", torch_model=True)
