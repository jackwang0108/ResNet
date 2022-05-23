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
    if pretrained:
        base_cmd += f"-p {pretrained} "

    base_cmd += f"-md {model} "
    base_cmd += f"-bz {bsize} "
    base_cmd += f"-ds {ds} "
    base_cmd += f"-m \"{message}\" "
    
    os.system(f"{base_cmd}")


if __name__ == "__main__":
    # for dataset in ["PascalVOC2012"]:
    #     for model in [50, 101, 152]:
    #         for torch_model in [False, True]:
    #             for skip in [False]:
    #                 for paper_train in [True]:
    #                     runs(
    #                         model=model, 
    #                         bsize=128 if model != 152 else 64, 
    #                         ds=dataset, 
    #                         torch_model=torch_model, 
    #                         paper_train=paper_train,
    #                         message=f"{'paper' if paper_train else 'modern'} train "\
    #                                 f"{'torch' if torch_model else 'my'} "\
    #                                 f"resnet{model} with{'' if skip else 'out'} skip", 
    #                     )

    for dataset in ["Cifar10", "Cifar100"]:
        if dataset == "Cifar10":
            continue
        for model in [18, 34, 50]:
            for torch_model in [False, True]:
                for skip in [False, True]:
                    for paper_train in [True, False]:
                        runs(
                            model=model, 
                            bsize=128, 
                            ds=dataset, 
                            torch_model=torch_model, 
                            paper_train=paper_train,
                            message=f"{'paper' if paper_train else 'modern'} train "\
                                    f"{'torch' if torch_model else 'my'} "\
                                    f"resnet{model} with{'' if skip else 'out'} skip", 
                        )


    # runs(model=34, bsize=128, ds="Cifar100", message="train my resnet34 without skip")
    # runs(model=34, bsize=128, ds="Cifar100", message="train my resnet34 without skip", torch_model=True)

    # runs(model=50, bsize=64, ds="Cifar100", message="train my resnet50 without skip")
    # runs(model=50, bsize=64, ds="Cifar100", message="train my resnet50 without skip", torch_model=True)
