# Bản của mình
import logging
import os
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule
from .sampling import split_and_shuffle
from tqdm import tqdm


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, transform=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)
        return {'image': image, 'label': label}


class ClsDataset(LightningDataModule):
    def __init__(self, data_dir, name='mnist',
                 train_split=0.8,
                 batch_size=32, train_shuffle=True,
                 width=224, height=224,
                 train_workers=4, val_workers=1
                ):
        super(ClsDataset, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.train_split = train_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_workers = train_workers
        self.val_workers = val_workers
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.width = width
        self.height = height
        self.num_classes = 0

    def setup(self, stage):
        if self.name == 'cifar10':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True,
                                        transform=apply_transform)
            test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True,
                                        transform=apply_transform)
            self.num_classes = 10
        elif self.name == 'cifar100':
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((self.width, self.height)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            raw_train_dataset = datasets.CIFAR100(self.data_dir, train=True, download=True,
                                            transform=apply_transform)
            test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True,
                                            transform=apply_transform)
            self.num_classes = 100
        else:
            raise NotImplementedError
        idx_train, idx_val = split_and_shuffle(raw_train_dataset, self.train_split) # Lấy index của các phần tử trong tập train gốc để cho vào train và val
        self.train_dataset = DatasetSplit(raw_train_dataset, idx_train) # Dựa vào index đã lấy, chia dữ liệu
        self.val_dataset = DatasetSplit(raw_train_dataset, idx_val)
        self.test_dataset = DatasetSplit(test_dataset, np.arange(len(test_dataset)))
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle,
                          num_workers=self.train_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.val_workers, pin_memory=True)
# Thêm phần drop last column cho train nếu cần