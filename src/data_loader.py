from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T


_TRANS = T.Compose([T.ToTensor()])


def get_dl(ds_name, path, train=True, batch_size=64):
    if ds_name == "mnist":
        data_set = MNIST(path + '/../data/mnist',  train=train, transform=_TRANS, download=True)
    else:
        raise Exception("Unsupported Dataset")

    return DataLoader(data_set, batch_size=batch_size, shuffle=True)