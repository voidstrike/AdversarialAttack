from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
import random
import torch
from PIL import Image


_TRANS = T.Compose([T.ToTensor()])


class ContrativeMNIST(Dataset):
    def __init__(self, dps, labels, transforms=None):
        self.items = dps
        self.labels = labels
        self.trans = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        d_tuple, labels = self.items[index], int(self.labels[index])
        d_tuple = (Image.fromarray(d_tuple[0].numpy(), mode='L'), Image.fromarray(d_tuple[1].numpy(), mode='L'))
        if self.trans is not None:
            d_tuple = (self.trans(d_tuple[0]), self.trans(d_tuple[1]))
        return d_tuple, labels


def get_dl(ds_name, path, train=True, batch_size=64):
    if ds_name == "mnist":
        data_set = MNIST(path + '/../data/mnist',  train=train, transform=_TRANS, download=True)
    else:
        raise Exception("Unsupported Dataset")

    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_cmnist_dl(path, train=True, batch_size=64):
    raw_data, raw_label = get_ds(path, train=train)
    CMNISTLoader = DataLoader(ContrativeMNIST(raw_data, raw_label, transforms=_TRANS), batch_size=batch_size, shuffle=True)
    return CMNISTLoader


def get_ds(path, train=True, max_len=1000):
    data_set = MNIST(path + '/../data/mnist',  train=train, transform=_TRANS, download=True)
    idx_set = [data_set.train_labels == k for k in range(10)]
    data_by_class = []
    for i in range(10):
        data_by_class.append(data_set.train_data[idx_set[i]])

    new_data = []
    new_label = []
    for c_class in range(10):
        c_len = len(data_by_class[c_class])
        for _ in range(max_len):
            new_data.append((data_by_class[c_class][random.randrange(c_len)],
                             data_by_class[c_class][random.randrange(c_len)]))
            new_label.append(1)
        o_class = list(range(0, c_class)) + list(range(c_class+1, 10))
        for _ in range(max_len):
            t_class = random.choice(o_class)
            t_len = len(data_by_class[t_class])
            new_data.append((data_by_class[c_class][random.randrange(c_len)],
                             data_by_class[t_class][random.randrange(t_len)]))
            new_label.append(0)

    return new_data, new_label

