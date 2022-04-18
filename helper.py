# Standard Library
import os
import pickle
import platform
from typing import *
from pathlib import PosixPath, WindowsPath, Path

# Third-party Library
import PIL
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, init
from terminaltables import SingleTable
from matplotlib import figure, axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

init(autoreset=True)

# Select System
system: str = platform.uname().system


# Decide Path
# Path: Union[PosixPath, WindowsPath]
# if system == "Windows":
#     Path = WindowsPath
# elif system == "Linux":
#     Path = PosixPath
# else:
#     raise NotImplementedError


class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    logs: Path = base.joinpath("logs")
    datasets: Path = base.joinpath("datasets")
    runs: Path = base.joinpath("runs")
    checkpoints: Path = base.joinpath("checkpoints")


for attr in ProjectPath.__dict__.values():
    if isinstance(attr, Path):
        attr.mkdir(parents=True, exist_ok=True)


class DatasetPath:
    class Cifar100:
        global system
        base: Path
        if system == "Windows":
            base = ProjectPath.datasets.joinpath("cifar100-windows")
        elif system == "Linux":
            base = ProjectPath.datasets.joinpath("cifar100-linux")
        meta: Path = base.joinpath("meta")
        test: Path = base.joinpath("test")
        train: Path = base.joinpath("train")

    class tinyImageNet:
        global system
        base: Path
        if system == "Windows":
            base = ProjectPath.datasets.joinpath("tinyimagenet-windows")
        elif system == "Linux":
            base = ProjectPath.datasets.joinpath("tinyimagenet-linux")
        train: Path = base.joinpath("train")
        val: Path = base.joinpath("val")
        test: Path = base.joinpath("test")
        wnids: Path = base.joinpath("wnids.txt")
        words: Path = base.joinpath("words.txt")

    def __str__(self):
        return "DatasetPath for iCaRL, containing Cifar100 and ImageNet2012"


# load label
with DatasetPath.Cifar100.meta.open(mode="rb") as f:
    meta: Dict[str, Any] = pickle.load(f)
cifar100_labels = meta["fine_label_names"]
cifar100_label2num = dict(zip(cifar100_labels, range(len(cifar100_labels))))
cifar100_num2label = dict(zip(cifar100_label2num.values(), cifar100_label2num.keys()))


# 多分类评价指标从confusion matrix中计算参考: https://zhuanlan.zhihu.com/p/147663370
class ClassificationEvaluator:
    """
    ClassificationEvaluator will automatically record each batch, calculate confusion matrix and give classification
    evaluations of top 1 or top 5.
    """

    def __init__(self, num_class: int) -> None:
        """
        init the evaluator

        Args:
            num_class (int): number of all prediction classes
        """
        self._num_class = num_class
        # rows: ground truth class, cols: predicted class
        self.top1_confusion_matrix = np.zeros(shape=(num_class, num_class))
        self.top5_confusion_matrix = np.zeros(shape=(num_class, num_class))

        # register measurement
        self._measure: Dict[str, Callable] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall
        }

    def add_batch_top1(self, y_pred: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        add_batch_top1 is used to record the confusion matrix of a batch to calculate top 5 evaluations

        Args:
            y_pred (Union[torch.Tensor, np.ndarray]): predictions made by the network, should be [batch, class_score]
            y (Union[torch.Tensor, np.ndarray]): ground truth label, should be [batch]

        Returns:
            np.ndarray: top 1 aconfusion matrix of the given batch

        Examples:
            >>> x, y = next(iter(dataloader))
            >>> y_pred = network(x)
            >>> ce = ClassificationEvaluator(num_class=100)
            >>> ce.add_batch_top1(y_pred)
            >>> # equals to
            >>> ce.add_batch_top1(y_pred.argmax(dim=1))
        """
        # check type
        assert isinstance(y_pred, (torch.Tensor, np.ndarray)), f"Not suppported type for pred_y: {type(y_pred)}"
        assert isinstance(y, (torch.Tensor, np.ndarray)), f"Not suppported type for y: {type(y)}"
        # check length
        assert (a := len(y_pred)) == (b := len(
            y)), f"None-equal predictions and ground truth, given prediction of {a} examples, but only with {b} ground truth"
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().to(device="cpu").numpy()
        y = y if isinstance(y, np.ndarray) else y.detach().to(device="cpu").numpy()

        # construc batch confusion matrix and add to self.confusion_matrix
        k = (y >= 0) & (y < self._num_class)

        # convert [batch, num_class] prediction scores to [batch] prediction results
        y_pred_cls = (y_pred if y_pred.ndim == 1 else y_pred.argmax(axis=1)).squeeze()

        confusion_matrix: np.ndarray
        # bincount for fast classification confusion matrix
        confusion_matrix = np.bincount(
            self._num_class * y.astype(int) + y_pred_cls.astype(int),
            minlength=self._num_class ** 2
        ).reshape(self._num_class, self._num_class)
        self.top1_confusion_matrix += confusion_matrix
        return confusion_matrix

    def add_batch_top5(self, y_pred: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """ 
        add_batch_top5 is used to record the confusion matrix of a batch to calculate the top 1 evaluations

        Args:
            y_pred (Union[torch.Tensor, np.ndarray]): prediction made by the network, should be [batch, class_score]
            y (Union[torch.Tensor, np.ndarray]): ground truth label, should be [batch]

        Returns:
            np.ndarray: top 1 confusion matrix of the given batch

        Examples:
            >>> x, y = next(iter(dataloader))
            >>> y_pred = network(x)
            >>> ce = ClassificationEvaluator(num_class=100)
            >>> ce.add_batch_top5(y_pred)
            >>> # the following will cause error
            >>> ce.add_batch_top5(y_pred.argmax(dim=1))
        """
        # check type
        assert isinstance(y_pred, (torch.Tensor, np.ndarray)), f"Not suppported type for pred_y: {type(y_pred)}"
        assert isinstance(y, (torch.Tensor, np.ndarray)), f"Not suppported type for y: {type(y)}"
        # check length
        assert (a := len(y_pred)) == (b := len(
            y)), f"None-equal predictions and ground truth, given prediction of {a} examples, but only with {b} ground truth"
        # check input
        assert y_pred.ndim == 2, f"For top5 evaluation, you should input [batch, class_score] tensor or ndarray, but you offered: {y_pred.shape}"
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.detach().to(device="cpu").numpy()
        y = y if isinstance(y, np.ndarray) else y.detach().to(device="cpu").numpy()

        # construc batch confusion matrix and add to self.confusion_matrix
        k = (y >= 0) & (y < self._num_class)

        # this could be done by torch.Tensor.topk, but for numpy, argsort is O(NlongN), following is a O(N) implementation
        # [1st, 2st, ..., 5st]
        y_pred_cls = np.argpartition(y_pred, kth=-5, axis=1)[:, -5:][:, ::-1]

        correct_mask = (y[:, np.newaxis] == y_pred_cls).any(axis=1)
        pred_yy = np.zeros_like(y)
        pred_yy[correct_mask] = y[correct_mask]
        pred_yy[~correct_mask] = y_pred_cls[~correct_mask, 0]

        confusion_matrix: np.ndarray
        # bincount for fast classification confusion matrix
        confusion_matrix = np.bincount(
            self._num_class * y.astype(int) + pred_yy.astype(int),
            minlength=self._num_class ** 2
        ).reshape(self._num_class, self._num_class)
        self.top5_confusion_matrix += confusion_matrix
        return confusion_matrix

    def accuracy(self, top: int = 5) -> np.float64:
        """
        accuracy calculate overall accuracy after an epoch

        Args:
            top (int, optional): Top k accuracy to calculate. Defaults to 5.

        Returns:
            np.float64: overall acuracy

        Examples:
            >>> mean_acc = ce.accuracy(top=1)
            >>> mean_acc
            0.8234321212
            >>> mean_acc = ce.accuracy(top=5)
            >>> mean_acc
            0.9312321245
        """
        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray = self.__getattribute__(f"top{top}_confusion_matrix")

        acc: np.ndarray = confusion_matrix.trace() / confusion_matrix.sum()
        return acc

    def precision(self, top: int = 5) -> Tuple[np.ndarray, np.float64]:
        """
        precision calculate per-class/mean precision after an epoch

        Args:
            top (int, optional): Top k precision to calculate. Defaults to 5.

        Returns:
            Tuple[np.ndarray, np.float64]: precisions of each class (np.ndarray) and mean precision (np.float64)

        Examples:
            >>> per_class_precision, mean_precision = ce.precision(top=1)
            >>> per_class_precision, mean_precision = ce.precision(top=5)
        """
        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray = self.__getattribute__(f"top{top}_confusion_matrix")

        # ignore zero division error, invalid division warning
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision: np.ndarray = confusion_matrix.diagonal() / confusion_matrix.sum(axis=0)
            per_class_precision[np.isnan(per_class_precision)] = 0
        mean_precision = per_class_precision.mean()
        return per_class_precision, mean_precision

    def recall(self, top: int = 5) -> Tuple[np.ndarray, np.float64]:
        """
        recall calculate per-class/mean recall after an epoch

        Args:
            top (int, optional): Top k recall to calculate. Defaults to 5.

        Returns:
            Tuple[np.ndarray, np.float64]: recall of each class (np.ndarray) and mean recall (np.float64)

        Examples:
            >>> per_class_recall, mean_recall = ce.precision(top=1)
            >>> per_class_recall, mean_recall = ce.precision(top=5)
        """
        assert top in [1, 5], f"Only support for top1 and top5"
        confusion_matrix: np.ndarray = self.__getattribute__(f"top{top}_confusion_matrix")

        # ignore zero division error, invalid division warning
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall: np.ndarray = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
            per_class_recall[np.isnan(per_class_recall)] = 0
        mean_recall = per_class_recall.mean()
        return per_class_recall, mean_recall

    def new_epoch(self) -> None:
        """
        new_epoch refresh all tracked batches, should be called in a new epoch

        Returns:
            None

        Examples:
            >>> for epoch in range(n_epoch):
            >>>     ce.new_epoch()
            >>>     # train
            >>>     for x, y in train_loader:
            >>>         # pass
            >>>         ce.add_batch_top1(...)
            >>>         ce.add_batch_top5(...)
        """
        self.top1_confusion_matrix = np.zeros(shape=(self._num_class, self._num_class))
        self.top5_confusion_matrix = np.zeros(shape=(self._num_class, self._num_class))

    def make_grid(self, title: str, labels: List[str], top: int = 5) -> str:
        """
        make_grid generate evaluation table after an epoch

        Args:
            title (str): title of the table
            labels (List[str]): all labels
            top (int, optional): select top k evaluations. Defaults to 5.

        Returns:
            str: generated table

        Examples:
            >>> print(ce.make_grid())
        """
        assert len(
            labels) == self._num_class, f"Evaluator is initialized with {self._num_class} classes, but reveive only {len(labels)} labels."
        data = []
        index = []
        column = labels + ["Mean"]
        last_col = []
        for measure_name, meansure in self._measure.items():
            index.append(measure_name)
            result = meansure(top=top)
            if isinstance(result, tuple):
                a = np.random.randn(10, 10)
                np.round(a, )
                data.append(np.round(result[0], decimals=4))
                last_col.append(np.round(result[1], decimals=4))
            else:
                data.append(np.array(["---"] * (len(column) - 1)))
                last_col.append(np.round(result, decimals=4))
        data = np.array(data)
        last_col = np.array(last_col)[:, np.newaxis]
        df = pd.DataFrame(np.hstack((data, last_col)), index=index, columns=column).T
        data = df.to_numpy()
        index = df.index.to_numpy()
        column = df.columns.tolist()
        data = np.hstack((index[:, np.newaxis], data)).tolist()
        column.insert(0, "class")
        data.insert(0, column)
        data[-1] = [f"{Fore.BLUE}{i}{Fore.RESET}" for i in data[-1]]
        table = SingleTable(data)
        table.title = title + f" top {top}"
        return table.table


def visualize(image: Union[torch.Tensor, np.ndarray],
              cls: Union[None, int, str, torch.Tensor, np.ndarray] = None) -> np.ndarray:
    """
    visualize given image(s) and return a grid of all given image(s) with label(s) (if provided)

    Args:
        image (Union[torch.Tensor, np.ndarray]): images to display, should be in the shape of [channel, width, height] or [batch, channel, width, height]
        cls (Union[None, int, str, torch.Tensor, np.ndarray], optional): label(s) of all given image(s). Defaults to None.

    Returns:
        np.ndarray: rendered iamges ([height, width, channel]), can be display by PIL or matplotlib

    Examples:
        >>> x, y = next(iter(train_loader))
        >>> visualize(image=x, cls=y)
    """
    num = 1 if image.ndim == 3 else image.shape[0]
    cls = np.array([""] * num) if cls is None else cls
    cls = np.array([cls]) if isinstance(cls, (int, str)) else cls
    cls = cls.numpy() if isinstance(cls, torch.Tensor) else cls
    try:
        assert num == len(cls), f"{num} images with {len(cls)} labels"
    except TypeError:
        cls = np.array([cls.item()])
    image: torch.Tensor = image if isinstance(image, torch.Tensor) else torch.from_numpy(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    assert image.shape[
               1] == 3, f"shape of image should be [batch_size, channel, width, height] or [channel, width, height]"
    image = image.permute(0, 2, 3, 1)

    cols = int(np.sqrt(num))
    rows = num // cols + (0 if num % cols == 0 else 1)

    if isinstance(cls[0], str):
        converter = lambda x: x
    else:
        converter = lambda x: cifar100_num2label[x]

    ax: List[axes.Axes]
    fig: figure.Figure
    fig, ax = plt.subplots(nrows=rows, ncols=cols, tight_layout=True, figsize=(1 * rows, 2 * cols))
    for i in range(rows):
        if rows == 1 and cols == 1:
            ax.imshow(image[i])
            ax.set_title(converter(cls[i]))
            ax.set_axis_off()
        elif cols == 1 and rows > 1:
            ax[i].imshow(image[i])
            ax[i].set_title(converter(cls[i]))
            ax[i].set_axis_off()
        else:
            for j in range(cols):
                idx = i * cols + j - 1
                if idx < num:
                    ax[i][j].imshow(image[idx])
                    ax[i][j].set_title(converter(cls[idx]))
                ax[i][j].set_axis_off()
    # plt.subplots_adjust()
    canvas = fig.canvas
    canvas.draw()

    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))


def legal_converter(path: Path) -> Path:
    """
    legal_converter convert path to legal path in different os

    Args:
        path (Path): path to convert

    Returns:
        path: legal path

    Examples:
        >>> p = Path("1:2:3.1.2.3")
        >>> legal_converter(p)
        1_2_3.1.2.3
    """
    global system
    if system == "Windows":
        illegal_char = ["<", ">", ":", "\"", "'", "/", "\\", "|", "?", "*"]
    elif system == "Linux":
        illegal_char = ["\\"]
    relative_path: List[str] = list(str(path.relative_to(ProjectPath.base)).split("\\"))
    for idx in range(len(relative_path)):
        for cc in illegal_char:
            relative_path[idx] = relative_path[idx].replace(cc, "_")
    return ProjectPath.base.joinpath(*relative_path)


if __name__ == "__main__":
    # check paths
    # dp = DatasetPath()
    # for p in dp.Cifar100.__dict__.values():
    #     if isinstance(p, Path):
    #         print(p, p.exists())
    # for p in dp.tinyImageNet.__dict__.values():
    #     if isinstance(p, Path):
    #         print(p, p.exists())

    # check labels
    # import pprint
    #
    # pprint.pprint(labels)
    # pprint.pprint(label2num)
    # pprint.pprint(num2label)

    # test legal_converter
    # import datetime
    # from network import ResNet34

    # lc = legal_converter(ProjectPath.runs / ResNet34.__name__ / str(datetime.datetime.now()))
    # print(lc)

    # test evaluator
    ce = ClassificationEvaluator(num_class=10)
    y = np.repeat(np.arange(0, 10)[np.newaxis, :], repeats=10).flatten()
    pred_y = np.zeros(shape=(100))
    # top1
    # ce.add_batch(pred_y=pred_y, y=y)
    # ce.add_batch_top1(y_pred=y, y=y)
    # top5
    pred_y = np.random.random(size=(100, 10))
    ce.add_batch_top1(y_pred=pred_y, y=y)
    ce.add_batch_top5(y_pred=pred_y, y=y)

    print(ce.accuracy(top=1))
    print(ce.accuracy(top=5))
    print(ce.precision(top=1))

    # print(ce.make_grid(title="Epoch 1", top=1, labels=cifar100_labels[:10]))
