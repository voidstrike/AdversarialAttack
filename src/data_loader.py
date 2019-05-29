from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.transforms as T
import numpy as np
import random
import torch
from PIL import Image


_TRANS = T.Compose([T.ToTensor()])
_TRANS_NORM = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])


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


class ContrastiveCIFAR(Dataset):
    def __init__(self, dps, labels, transforms=None):
        self.items = dps
        self.labels = labels
        self.trans = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        d_tuple, labels = self.items[index], int(self.labels[index])
        d_tuple = (Image.fromarray(d_tuple[0]), Image.fromarray(d_tuple[1]))
        if self.trans is not None:
            d_tuple = (self.trans(d_tuple[0]), self.trans(d_tuple[1]))
        return d_tuple, labels


def get_dl(ds_name, path, train=True, batch_size=64):
    if ds_name == "mnist":
        data_set = MNIST(path + '/../data/mnist',  train=train, transform=_TRANS, download=True)
    elif ds_name == 'cifar':
        data_set = CIFAR10(path + '/../data/cifar10', train=train, transform=_TRANS_NORM, download=True)
    else:
        raise Exception("Unsupported Dataset")

    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_dl_w_sampler(ds_name, path, train=True, batch_size=64, sample_size=1000):
    if ds_name == 'mnist':
        data_set = MNIST(path + '/../data/mnist', train=train, transform=_TRANS, download=True)
    elif ds_name == 'cifar':
        data_set = CIFAR10(path + '/../data/cifar10', train=train, transform=_TRANS_NORM, download=True)
    else:
        raise Exception("Unsupported Dataset")

    start_index = random.randrange(0, 10000 - sample_size)
    sampler = SubsetRandomSampler(list(range(start_index, start_index + sample_size)))
    return DataLoader(data_set, batch_size=batch_size, shuffle=False, sampler=sampler)


def get_cmnist_dl(path, train=True, batch_size=64):
    raw_data, raw_label = get_ds(path, train=train)
    CMNISTLoader = DataLoader(ContrativeMNIST(raw_data, raw_label, transforms=_TRANS), batch_size=batch_size, shuffle=True)
    return CMNISTLoader


def get_ccifar_dl(path, train=True, batch_size=64):
    raw_data, raw_label = get_ds(path, dname='cifar', train=train)
    CCIFARLoader = DataLoader(ContrastiveCIFAR(raw_data, raw_label, transforms=_TRANS_NORM), batch_size=batch_size, shuffle=True)
    return CCIFARLoader


def get_ds(path, dname='mnist', train=True, max_len=1000):
    if dname == 'mnist':
        data_set = MNIST(path + '/../data/mnist',  train=train, transform=_TRANS, download=True)
        idx_set = [data_set.train_labels == k for k in range(10)]
    elif dname == 'cifar':
        data_set = CIFAR10(path + '/../data/cifar10', train=train, transform=_TRANS_NORM, download=True)
        idx_set = [[] for _ in range(10)]
        for i, label in enumerate(data_set.train_labels):
            idx_set[label].append(i)
    else:
        raise Exception('Unsupported Dataset')

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
            new_label.append(0)
        o_class = list(range(0, c_class)) + list(range(c_class+1, 10))
        for _ in range(max_len):
            t_class = random.choice(o_class)
            t_len = len(data_by_class[t_class])
            new_data.append((data_by_class[c_class][random.randrange(c_len)],
                             data_by_class[t_class][random.randrange(t_len)]))
            new_label.append(1)

    return new_data, new_label

