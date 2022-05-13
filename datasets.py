
# Standard Library
import pickle
from typing import *
from pathlib import Path

# Third-party Party
import numpy as np
import PIL.Image as Image
from colorama import Fore, init

# Torch Library
import torch
import torch.utils.data as data
import torchvision.transforms as T

# My Library
from helper import visualize_np, visualize_plt, visualize_pil
from helper import ProjectPath, DatasetPath
from helper import ClassLabelLookuper

init(autoreset=True)

ImageType = TypeVar(
    "ImageType",
    np.ndarray, torch.Tensor, Path
)

ClassType = TypeVar(
    "ClassType",
    np.ndarray, torch.Tensor
)


class MultiDataset(data.Dataset):
    def __init__(self, dataset: str, split: str):
        super(MultiDataset, self).__init__()
        assert split in (s := ["train", "val", "test"]), f"{Fore.RED}Invalid split, should be in {s}"
        self.split = split
        self.dataset = dataset
        self._dataset_reader: Dict[str, Callable] = {
            "Cifar10": self.__read_cifar10,
            "Cifar100": self.__read_cifar100,
            "PascalVOC2012": self.__read_PascalVOC2012
        }
        assert dataset in self._dataset_reader.keys(), f"{Fore.RED}Invalid dataset, please select in " \
                                                       f"{self._dataset_reader.keys()}."
        self.image: Union[np.ndarray, List[Path]]
        self.label: np.ndarray
        self.image, self.label = self._dataset_reader[self.dataset]()
        self.select_train_val()
        self.num_class = len(ClassLabelLookuper(self.dataset).cls)

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.image[idx], self.label[idx]
        if isinstance(image, Path):
            image = Image.open(image)
        else:
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        return self.transform(image), label
    
    def set_transform(self, transform: T.Compose) -> "MultiDataset":
        self.transform = transform
        return self

    def select_train_val(self, trainval_ratio: Optional[float] = 0.2) -> None:
        # get image of each label
        self.label_image: Dict[int, np.ndarray] = {}
        for label in np.unique(self.label):
            self.label_image[label] = np.where(self.label == label)[0]

        if self.dataset in ["Cifar10", "Cifar100"]:
            if self.split == "test":
                return
            else:
                # generate train val if not exists, else load
                if (config_path := ProjectPath.config.joinpath(f"{self.dataset}.npz")).exists():
                    data = np.load(config_path)
                    ratio, train, val =data["ratio"], data["train"], data["val"]
                if not config_path.exists() or ratio != trainval_ratio:
                    train, val = [], []
                    for label, image_idx in self.label_image.items():
                        np.random.shuffle(image_idx)
                        val_num = int(trainval_ratio * len(image_idx))
                        val.append(image_idx[:val_num])
                        train.append(image_idx[val_num:])
                    train = np.stack(train, axis=0)
                    val = np.stack(val, axis=0)
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(config_path, ratio=trainval_ratio, train=train, val=val)
                train = np.concatenate(train, axis=0)
                val = np.concatenate(val, axis=0)
                
                # select train val
                if self.split == "val":
                    self.image = self.image[val]
                    self.label = self.label[val]
                else:
                    self.image = self.image[train]
                    self.label = self.label[train]
        else:
            return


    def __read_cifar10(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.split in ["train", "val"]:
            data = []
            for batch in DatasetPath.Cifar10.train:
                with batch.open(mode="rb") as f:
                    data.append(pickle.load(f, encoding="bytes"))
            image = np.concatenate([i[b"data"].reshape(-1, 3, 32, 32) for i in data], axis=0)
            label = np.concatenate([i[b"labels"] for i in data], axis=0)
        else:
            with DatasetPath.Cifar10.test.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data[b"data"].reshape(-1, 3, 32, 32)
            label = data[b"labels"]
        return image.transpose(0, 2, 3, 1), np.array(label)

    def __read_cifar100(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.split in ["train", "val"]:
            with DatasetPath.Cifar100.train.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data[b"data"].reshape(-1, 3, 32, 32)
            label = data[b"fine_labels"]
        else:
            with DatasetPath.Cifar100.test.open(mode="rb") as f:
                data = pickle.load(f, encoding="bytes")
            image = data["data"].reshape(-1, 3, 32, 32)
            label = data["label"]
        return image.transpose(0, 2, 3, 1), np.asarray(label)

    def __read_PascalVOC2012(self) -> Tuple[List[Path], np.ndarray]:
        image = []
        label = []
        ccn = ClassLabelLookuper(datasets="PascalVOC2012")
        pascalvoc2012 = DatasetPath.PascalVOC2012.classification()
        if self.split in "train":
            for k, v in pascalvoc2012.train.items():
                image.extend(v)
                label.extend([ccn.get_label(k)] * len(v))
        elif self.split == "val":
            for k, v in pascalvoc2012.val.items():
                image.extend(v)
                label.extend([ccn.get_label(k)] * len(v))
        else:
            assert False, f"{Fore.RED}PascalVOC2012 test data is not accesibly"
        # Attention: PascalVOC2012 中图像是存在重复的
        image, idx = np.unique(image, return_index=True)
        return image, np.array(label)[idx]

if __name__ == "__main__":
    # md = MultiDataset(dataset="PascalVOC2012", split="val")
    # tt = T.Compose([
    #     T.RandomHorizontalFlip(),
    #     T.Resize((224, 224)),
    #     T.ToTensor()
    # ])
    # md.set_transform(tt)

    md = MultiDataset(dataset="PascalVOC2012", split="train")
    tt = T.Compose([
        T.Resize(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    md.set_transform(tt)

    ccn = ClassLabelLookuper(datasets=md.dataset)
    dl = data.DataLoader(md, batch_size=64)
    for x, y in dl:
        print(x.shape)
        visualize_pil(x, [ccn.get_class(i.item()) for i in y]).show()
        break

