# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# standard libraries
import datetime
from pathlib import Path

# third-party libraries
import tqdm
import numpy as np
from colorama import Fore, init

# my libraries
from network import ResNet34
from dataset import Cifar100
from helper import ProjectPath, DatasetPath, ClassificationEvaluator, visualize, legal_converter, cifar100_labels, cifar100_num2label, cifar100_label2num, system


class FullTrainer:
    start_time = str(datetime.datetime.now())

    num_workers: int = 0 if system == "Windows" else 1
    available_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    default_dtype = torch.float

    def __init__(self, network: nn.Module, dry_run: bool=True):
        """
        __init__ the trainer of ResNet

        Args:
            network (nn.Module): ResNet instance, should be subclass of _ResNetBas
            dry_run (bool, optional): if true, no checkpoints or run files will be generate. Defaults to True.
        """
        # network modules
        self.network = network.to(dtype=self.default_dtype, device=self.available_device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optim = optim.SGD(params=self.network.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)

        # summary writer
        suffix = self.network.__class__.__name__
        self.dry_run: bool = dry_run
        if not dry_run:
            writer_path = legal_converter(ProjectPath.runs / suffix / self.start_time)
            self.writer = SummaryWriter(log_dir=writer_path)

        # datasets
        self.train_set = Cifar100(split="train")
        self.val_set = Cifar100(split="val")
        self.test_set = Cifar100(split="test") 


        assert (a:=network.target_dataset) == (b:=self.train_set.__name__), f"Incooperate network and dataset, network is for {a}, but dataset is {b}"

        # checkpoint paths
        self.checkpoint_path = legal_converter(ProjectPath.checkpoints / suffix / self.start_time / "best.pt")
        if not dry_run:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # evaluator
        self.val_evaluator = ClassificationEvaluator(num_class=network.num_class)


    def __del__(self):
        if not self.dry_run:
            self.writer.close()

    def train(self, n_epoch: int, early_stop: int = 200, plateau: int = 30, show_grid: bool = True) -> nn.Module:
        """
        train the network

        Args:
            n_epoch (int): total epoch
            early_stop (int, optional): early stop counts. Defaults to 200.
            plateau (int, optional): number of epochs to decrease learning rate when accuracy reaches plateau. Defaults to 50.
            show_grid (bool, optional): if print table in the terminal. Defaults to True.

        Returns:
            nn.Module: network with trained parameter
        """

        # loaders
        train_loader = data.DataLoader(
            self.train_set, batch_size=128, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        val_loader = data.DataLoader(
            self.val_set, batch_size=128, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        maximum_acc: float = 0
        early_stop_cnt: int = 0

        x: torch.Tensor
        y: torch.Tensor
        loss: torch.Tensor
        for epoch in (tbar := tqdm.trange(n_epoch)):
            ndigit = len(str(n_epoch))
            tbar.set_description(f"Epoch [{Fore.GREEN}{epoch:>{ndigit}d}{Fore.RESET}/{n_epoch:>{ndigit}d}]")
            self.train_evaluator.new_epoch()
            self.val_evaluator.new_epoch()
            # train
            for step, (x, y) in enumerate(train_loader):
                # do not change y from long to float
                # the following line takes about 84.7% time to run, so it's better to set pin_memory and non_blocking to True
                x = x.to(dtype=self.default_dtype)
                if self.available_device != "cpu":
                    x, y = x.cuda(), y.cuda()
                # !!! don't forget
                self.optim.zero_grad()
                y_pred = self.network(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optim.step()
                

            # val
            with torch.no_grad():
                for step, (x, y) in enumerate(val_loader):
                    x, y = x.to(dtype=self.default_dtype, device=self.available_device), y.to(self.available_device)
                    y_pred = self.network(x)
                    
                    # log
                    self.val_evaluator.add_batch_top1(y_pred=y_pred, y=y)
                    self.val_evaluator.add_batch_top5(y_pred=y_pred, y=y)

            # log
            top1_acc = self.val_evaluator.accuracy(top=1)
            top5_acc = self.val_evaluator.accuracy(top=5)
            top1_precision = self.val_evaluator.precision(top=1)[1]
            top5_precision = self.val_evaluator.precision(top=5)[1]
            top1_recall = self.val_evaluator.recall(top=1)[1]
            top5_recall = self.val_evaluator.recall(top=5)[1]

            if not self.dry_run:
                self.writer.add_scalar(tag="acc/top1", scalar_value=top1_acc, global_step=epoch)
                self.writer.add_scalar(tag="acc/top5", scalar_value=top5_acc, global_step=epoch)
                self.writer.add_scalar(tag="precision/top1", scalar_value=top1_precision, global_step=epoch)
                self.writer.add_scalar(tag="precision/top5", scalar_value=top5_precision, global_step=epoch)
                self.writer.add_scalar(tag="recall/top1", scalar_value=top1_recall, global_step=epoch)
                self.writer.add_scalar(tag="recall/top5", scalar_value=top5_recall, global_step=epoch)


            tbar.write(
                f"Epoch [{epoch:>{ndigit}d}/{n_epoch:>{ndigit}d}], "
                f"top1 acc = {top1_acc:>.{ndigit}f}, "
                f"top5 acc = {top5_acc:>.{ndigit}f}, "
                f"top5 precision = {top5_precision:>.{ndigit}f}, "
                f"top5 recall = {top5_recall:>.{ndigit}f}"
            )



            # early stop
            if maximum_acc < (new_acc:=top5_acc):
                maximum_acc = new_acc
                early_stop_cnt = 0
                # print log
                if show_grid:
                    tbar.write(self.val_evaluator.make_grid(title=f" Epoch {epoch}", labels=cifar100_labels, top=5))
                else:
                    tbar.write(f"{Fore.BLUE}Maximum accuracy: {maximum_acc}")
                if not self.dry_run:
                    tbar.write(f"{Fore.BLUE}Save checkpoints")
                    torch.save(self.network.state_dict(), self.checkpoint_path)
            else:
                early_stop_cnt += 1

            # decrease lr
            if early_stop_cnt >= plateau:
                tbar.write(f"{Fore.CYAN}Decreased learning rate")
                for g in self.optim.param_groups:
                    g["lr"] /= 10

            if early_stop_cnt >= early_stop:
                tbar.write(f"{Fore.CYAN}Early Stop!")
                tbar.close()
                break
        return self.network
    
    @torch.no_grad()
    def test(self):
        test_loader = data.DataLoader(
            self.test_set, batch_size=128, shuffle=False, num_workers=self.num_workers
        )



if __name__ == "__main__":
    resnet34 = ResNet34(num_class=100, target_dataset="Cifar100")
    trained_resnet34 = FullTrainer(network=resnet34, dry_run=False).train(n_epoch=1000, early_stop=200, show_grid=False)

    # cprofile
    # ft = FullTrainer(network=resnet34, dry_run=True).train(n_epoch=1, show_grid=False)

    # from line_profiler import LineProfiler
    # line profile
    # lp = LineProfiler()
    # train = lp(FullTrainer(network=resnet34, dry_run=True).train)
    # train(n_epoch=1, show_grid=False)
    # lp.print_stats()
    # print("Done")

