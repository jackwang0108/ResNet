# Standard Library
import pickle
from typing import *
from pathlib import Path
from dataclasses import dataclass

# Third-Party Library
import torch
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

init(autoreset=True)


@dataclass
class ProjectPath:
    base: Path = Path(__file__).resolve().parent
    runs: Path = base.joinpath("runs")
    log: Path = base.joinpath("logs")
    config: Path = base.joinpath("config")
    dataset: Path = base.joinpath("datasets")
    checkpoints: Path = base.joinpath("checkpoints")

    def __init__(self) -> None:
        for project_path in ProjectPath.__dict__.values():
            if isinstance(project_path, Path):
                project_path.mkdir(parents=True, exist_ok=True)


PascalClassificationType = Dict[str, List[Path]]
PascalSegmentationType = List[Path]
PascalType = TypeVar("PascalType", PascalClassificationType, PascalSegmentationType)


class DatasetPath:
    base: Path = ProjectPath.dataset

    class Cifar10:
        base: Path = ProjectPath.base.joinpath("datasets/cifar-10")
        meta: Path = base.joinpath("batches.meta")
        test: Path = base.joinpath("test_batch")
        train: List[Path] = list(base.glob("data*"))

    class Cifar100:
        base: Path = ProjectPath.base.joinpath("datasets/cifar-100")
        meta: Path = base.joinpath("meta")
        test: Path = base.joinpath("test")
        train: Path = base.joinpath("train")

    class PascalVOC2012:
        base: Path = ProjectPath.base.joinpath("datasets/PascalVOC2012").resolve()
        JPEGImages: Path = base.joinpath("JPEGImages")
        ImageSets: Path = base.joinpath("ImageSets")
        Annotation: Path = base.joinpath("Annotation")
        SegmentationClass: Path = base.joinpath("SegmentationClass")
        SegmentationObject: Path = base.joinpath("SegmentationObject")

        train: PascalType
        val: PascalType

        @classmethod
        def classification(cls) -> "PascalVOC2012":
            # Attention: 同一个图像是存在多个标签的, 后续需要去重复
            train_idx: Dict[str, List[Path]] = {}
            train_class_idx: List[Path] = list(cls.ImageSets.joinpath("Main").glob(r"*_train.txt"))

            # get train
            seen_img = np.array(["-1"], dtype=str)
            for path in train_class_idx:
                clss = path.stem.split("_")[0]
                data = np.loadtxt(path, dtype=str)
                mask = np.where(data == "1")[0]
                train_idx[clss] = np.apply_along_axis(
                    arr=data[mask], axis=1,
                    func1d=lambda x: cls.JPEGImages.joinpath(f"{x[0]}.jpg")
                ).tolist()

            # get validation
            val_idx: Dict[str, List[Path]] = {}
            val_class_idx: List[Path] = list(cls.ImageSets.joinpath("Main").glob(r"*_val.txt"))
            for path in val_class_idx:
                clss = path.stem.split("_")[0]
                data = np.loadtxt(path, dtype=str)
                mask = np.where(data == "1")[0]
                val_idx[clss] = np.apply_along_axis(
                    arr=data[mask], axis=1,
                    func1d=lambda x: cls.JPEGImages.joinpath(f"{x[0]}.jpg")
                ).tolist()

            # add to class property
            cls.train: PascalClassificationType = train_idx
            cls.val: PascalClassificationType = val_idx
            return cls

        @classmethod
        def segmentation(cls) -> "PascalVOC2012":
            seg_path = cls.ImageSets.joinpath("Segmentation")
            train = np.loadtxt(seg_path.joinpath("train.txt"), dtype=str)[:, np.newaxis]
            val = np.loadtxt(seg_path.joinpath("val.txt"), dtype=str)[:, np.newaxis]

            cls.train: PascalSegmentationType = np.apply_along_axis(
                arr=train, axis=1,
                func1d=lambda x: cls.SegmentationClass.joinpath(f"{x[0]}.png")
            )
            cls.val: PascalSegmentationType = np.apply_along_axis(
                arr=val, axis=1,
                func1d=lambda x: cls.SegmentationClass.joinpath(f"{x[0]}.png")
            )

            return cls


class ClassLabelLookuper:
    def __init__(self, datasets: str) -> None:
        assert datasets in (s := [name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)]), \
            f"{Fore.RED}Invalid Dataset, should be in {s}, but you offered {datasets}"

        self.cls: List[str]
        self._cls2label: Dict[str, int]
        self._label2cls: Dict[int, str]

        if datasets == "Cifar10":
            with DatasetPath.Cifar10.meta.open(mode="rb") as f:
                meta = pickle.load(f)
            self.cls = meta["label_names"]
        elif datasets == "Cifar100":
            with DatasetPath.Cifar100.meta.open(mode="rb") as f:
                meta = pickle.load(f)
            self.cls = meta["fine_label_names"]
        else:
            self.cls = DatasetPath.PascalVOC2012.classification().train.keys()

        self._cls2label = dict(zip(self.cls, range(len(self.cls))))
        self._label2cls = dict(zip(range(len(self.cls)), self.cls))

    def get_class(self, label: int) -> str:
        return self._label2cls[label]

    def get_label(self, cls: str) -> int:
        return self._cls2label[cls]


class ClassificationEvaluator:
    def __init__(self, dataset: str):
        self._ccn: ClassLabelLookuper = ClassLabelLookuper(datasets=dataset)
        self.ds = dataset
        self.cls: List[str] = self._ccn.cls
        self.confusion_top1 = np.zeros(shape=(len(self.cls),) * 2, dtype=int)
        self.confusion_top5 = np.zeros(shape=(len(self.cls),) * 2, dtype=int)

    def __check(self, top: int):
        assert top in [1, 5], f"{Fore.RED}Wrong top-k, can only be top 1 or top 5"
    
    def new_epoch(self):
        self.confusion_top1 = np.zeros(shape=(len(self.cls),) * 2, dtype=int)
        self.confusion_top5 = np.zeros(shape=(len(self.cls),) * 2, dtype=int)
    
    def record(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().topk(k=5, dim=1, largest=True, sorted=True)[1].numpy()

        # get valid examples
        k = (y >= 0) & (y < len(self.cls))

        # get top 5 predictions
        # Warn: 下面这句话计算的有问题, 直接使用torch.topk计算
        # y_pred = np.argpartition(y_pred, kth=-5, axis=1)[:, -5:][:, ::-1]

        # get top 1 confusion matrix of the batch
        top1_cm = np.bincount(
            len(self.cls) * y[k].astype(int) + y_pred[k, 0].astype(int),
            minlength=len(self.cls) ** 2
        ).reshape((len(self.cls), ) * 2)
        self.confusion_top1 += top1_cm

        # get top 5 confusion matrix of the batch
        correct_mask = (y[:, np.newaxis] == y_pred).any(axis=1)
        _y_pred = np.zeros_like(y)
        _y_pred[correct_mask] = y[correct_mask]
        _y_pred[~correct_mask] = y_pred[~correct_mask, 0]
        y_pred = _y_pred

        top5_cm = np.bincount(
            len(self.cls) * y[k].astype(int) + y_pred[k].astype(int),
            minlength=len(self.cls) ** 2
        ).reshape((len(self.cls), ) * 2)
        self.confusion_top5 += top5_cm

    def get_confusion(self, top: Optional[int] = 1, title: Optional[str] = None, tofile: bool = False) -> Union[str, pd.DataFrame]:
        self.__check(top)
        confusion_matrix = getattr(self, f"confusion_top{top}")
        df = pd.DataFrame(confusion_matrix, index=self.cls, columns=self.cls)
        df.index.name = "gt"
        df.columns.name = "pred"
        try:
            no_tbs = False
            from terminaltables import AsciiTable
            ll = df.reset_index().T.reset_index().T.values.tolist()
            ll[0][0] = u"gt\u21A1 |pred\u21A0 "
            table = AsciiTable(ll)
            for i in range(len(self.cls) + 1):
                table.justify_columns[i] = "center"
            if table.ok or tofile:
                table = str(table.table).split("\n")
                len_row = len(table[0])
                c = f"Top {top} Confusion Matrix of dataset {self.ds}" if title is None else title
                table[0] = "+" + c.center(len_row - 2, "-") + "+"
                t = "\n".join(table)
                return t
        except ModuleNotFoundError:
            s1 = f"{Fore.YELLOW}terminaltables not found, print with pd. " \
                 f"If you want prettier output, please install terminaltables"
            no_tbs = True
        s2 = f"{Fore.YELLOW}Terminal table break detected, print with pd"
        pd.options.display.max_columns = len(self.cls)
        pd.options.display.max_rows = len(self.cls)
        print(s1 if no_tbs else s2)
        return df
    
    @property
    def acc(self) -> Tuple[float, float]:
        accs = []
        for top in [1, 5]:
            cm = getattr(self, f"confusion_top{top}")
            with np.errstate(divide='ignore', invalid='ignore'):
                accs.append(
                    np.nan_to_num(cm.trace() / cm.sum())
                )
        return tuple(accs)

    @property
    def recall(self) -> Tuple[float, float]:
        recalls = []
        for top in [1, 5]:
            cm = getattr(self, f"confusion_top{top}")
            with np.errstate(divide="ignore", invalid="ignore"):
                per_class_recall: np.ndarray = cm.diagonal() / cm.sum(axis=1)
                per_class_recall[np.isnan(per_class_recall)] = 0
                mean_recall = per_class_recall.mean()
            recalls.append(mean_recall.item())
        return tuple(recalls)

    @property
    def precision(self) -> Tuple[float, float]:
        precisions = []
        for top in [1, 5]:
            cm = getattr(self, f"confusion_top{top}")
            with np.errstate(divide="ignore", invalid="ignore"):
                per_class_precision: np.ndarray = cm.diagonal() / cm.sum(axis=0)
                per_class_precision[np.isnan(per_class_precision)] = 0
                mean_precision = per_class_precision.mean()
            precisions.extend(mean_precision.item())
        return tuple(precisions)


ImageType = TypeVar(
    "ImageType",
    np.ndarray, torch.Tensor, Image.Image,
    List[np.ndarray], List[torch.Tensor], List[Image.Image]
)

ClassType = TypeVar(
    "ClassType",
    str,
    List[np.ndarray], List[torch.Tensor], List[Image.Image]
)


def _get_image(return_png: bool = False, driver: str = "ndarray"):
    assert driver in ["pil", "ndarray"]

    def visualize_func_decider(show_func: Callable = _visualize) -> Callable:
        def show_with_png(*args, **kwargs):
            show_func(*args, **kwargs)
            import matplotlib.backends.backend_agg as bagg
            canvas = bagg.FigureCanvasAgg(plt.gcf())
            canvas.draw()
            png, (width, height) = canvas.print_to_buffer()
            png = np.frombuffer(png, dtype=np.uint8).reshape(
                (height, width, 4))

            if driver == 'pil':
                return Image.fromarray(png)
            else:
                return png

        if return_png:
            return show_with_png
        else:
            return show_func

    return visualize_func_decider


def _visualize(image: ImageType, cls: Optional[ClassType] = None) -> None:
    image_list: List[np.ndarray]
    title_list: List[str]

    # type check
    assert isinstance(
        image, (Image.Image, np.ndarray, torch.Tensor, list)
    ), f"{Fore.RED}Wrong type, input type of image should be (Image.Image, np.ndarray, torch.Tensor, list). " \
       f"But received {type(image)}"
    if isinstance(image, list):
        assert all(
            isinstance(i, (Image.Image, np.ndarray, torch.Tensor, list)) for i in image
        ), f"{Fore.RED}Wrong type, input image type in the list should be (Image.Image, np.ndarray, torch.Tensor, " \
           f"list). But not all image in the image are valid"

    assert isinstance(cls,
                      (str, list)) or cls is None, f"{Fore.RED}Wrong type, input type of cls should be (str, list). " \
                                                   f"But received {type(cls)}"
    if isinstance(cls, list):
        assert all(
            isinstance(i, str) for i in cls
        ), f"{Fore.RED}Worng type, input cls in the list type should be str, but not all cls in the cls are valid"

    # make image
    image_list = []
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if isinstance(image, (np.ndarray, torch.Tensor)):
        assert image.ndim in [2, 3, 4], \
            f"{Fore.RED}Wrong shape, input dimension should be " \
            f"2: [height, width] for single 1-channel gray image, " \
            f"3: [height, width, channel] for single 3-channel color image, or multiple gray image and " \
            f"4: [batch, height, width, channel] for multiple 3-channel color image"
        image = image if isinstance(image, np.ndarray) else image.detach().cpu().numpy()
        # gray image
        if image.ndim == 2:
            image_list.append(np.expand_dims(image, axis=-1))
        elif image.ndim == 3:
            if (c_num := image.shape[-1]) == 3:
                # single 3-channel colored image [height, width, channel]
                image_list.append(image)
            elif (c_num := image.shape[0]) == 3:
                # single 3-channel colored image [channel, height, width]
                image_list.append(image.transpose(1, 2, 0))
            else:
                # multiple 1-channel gray image
                image_list.extend([image[..., i] for i in range(c_num)])
        else:
            # multiple 3-channel color image
            if image.shape[1] == 3:
                image = image.transpose(0, 2, 3, 1)
            image_list.extend([image[i, ...] for i in range(image.shape[0])])
    else:
        for img in image:
            if isinstance(img, Image.Image):
                image = np.asarray(img)
            if isinstance(img, (np.ndarray, torch.Tensor)):
                assert img.ndim in [2, 3], f"{Fore.RED}Wrong shape, input dimension should be " \
                                           f"2: [height, width] for single 1-channel gray image, " \
                                           f"3: [height, width, channel] for 3-channel color image"
                img = img if isinstance(img, np.ndarray) else img.detach().cpu().numpy()
                if img.ndim == 2:
                    image_list.append(np.expand_dims(img, axis=-1))
                elif img.ndim == 3:
                    assert img.shape[-1] == 3, f"{Fore.RED}Wrong shape, input image in the list should be 3-channel " \
                                               f"image"
                    image_list.append(img)

    # make title
    num_image = len(image_list)
    title_list = cls if isinstance(cls, list) else ["" if cls is None else cls] * num_image
    assert (num_cls := len(title_list)) == num_image or not isinstance(cls, list), \
        f"{Fore.RED}Image num and class num mismatch, {num_cls} images with {num_image} class"

    # draw
    import math
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

    n_row: int = int(math.sqrt(num_image))
    n_col: int = math.ceil(num_image / n_row)

    ax: List[List[Axes]]
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col, figsize=(2 * n_row, 2 * n_col))
    fig.tight_layout()
    if len(image_list) == 1:
        ax = [[ax]]
    elif n_row == 1:
        ax = [ax]

    fill_num = n_col * n_row - len(image_list)
    image_list.extend([np.ones(shape=(10, 10, 3), dtype=int) * 255] * fill_num)
    title_list.extend([""] * fill_num)

    for img_idx, (img, title) in enumerate(zip(image_list, title_list)):
        row = img_idx // n_col
        col = img_idx % n_col
        ax[row][col].imshow(img)
        ax[row][col].set_title(title)
        ax[row][col].set_axis_off()

    canvas = Canvas(fig)
    canvas.draw()


def visualize_plt(*args, **kwargs) -> None:
    _visualize(*args, **kwargs)
    plt.ion()
    plt.show()


def visualize_np(*args, **kwargs) -> np.ndarray:
    img = _get_image(return_png=True, driver="ndarray")(_visualize)(*args, **kwargs)
    plt.ion()
    plt.show()
    return img


def visualize_pil(*args, **kwargs) -> Image.Image:
    return _get_image(return_png=True, driver="pil")(_visualize)(*args, **kwargs)


if __name__ == "__main__":
    import pprint

    # pp = ProjectPath()
    # print(DatasetPath.Cifar10.train)
    # print(DatasetPath.Cifar100.train)
    # print(DatasetPath.PascalVOC2012.train_idx.keys().__len__())
    # print(type(DatasetPath), isinstance(DatasetPath, type))
    # print([name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)])
    # print(ClassLabelLookuper(datasets="Cifar10"))
    # print(ClassLabelLookuper(datasets="Cifar100"))

    # for i in [name for name, value in DatasetPath.__dict__.items() if isinstance(value, type)]:
    #     print(ClassLabelLookuper(i)._cls2label)

    # import pickle
    #
    # with DatasetPath.Cifar100.train.open(mode="rb") as f:
    #     data = pickle.load(f, encoding="bytes")
    #     images, labels = data[b"data"], data[b"fine_labels"]
    #     images = images.reshape(-1, 3, 32, 32)
    # ccn = ClassLabelLookuper(datasets="Cifar100")
    # length = 64
    # visualize_pil(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]]).show()
    # visualize_plt(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]])
    # a = visualize_np(images[0: length + 1], [ccn.get_class(i) for i in labels[0: length + 1]])
    # print(a.shape)

    # def rand_shape(): return (np.random.randint(100, 256), np.random.randint(100, 256))
    # image = [np.random.randint(low=0, high=256, size=(*rand_shape(), 3), dtype=int) for i in range(64)]
    # visualize_plt(image)

    # ce = ClassificationEvaluator(dataset="Cifar100")
    # ce.confusion_top1 += 10
    # a = ce.get_confusion()
    # print(a)

    # print(DatasetPath.PascalVOC2012.classification().train)
    for i in DatasetPath.PascalVOC2012.segmentation().train:
        print(i, i.exists())
    # print(DatasetPath.PascalVOC2012.segmentation().train)
