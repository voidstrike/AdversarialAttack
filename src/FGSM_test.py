import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.datasets import MNIST

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from Model import LeNetAE28
from data_loader import get_dl
import numpy as np
import sys
import os

trans = T.Compose([T.ToTensor()])
reverse_trans = lambda x: np.asarray(T.ToPILImage()(x))

eps = 2 * 8 / 225
steps = 40
norm = float('inf')
step_alpha = 1e-4


def getHitCount(t_label, p_label):
    _, p_label = p_label.max(1)
    num_correct = (t_label == p_label).sum().item()
    return num_correct


def train_model(ann, dl):
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(ann.parameters(), lr=1e-4)
    instance_count = dl.dataset.__len__()

    for step in range(100):
        loss_ce_iter, loss_acc_iter = .0, .0
        for features, labels in dl:
            if torch.cuda.is_available():
                features = Variable(features.view(features.shape[0], -1).cuda())
                labels = Variable(labels.cuda())

            else:
                features = Variable(features.view(features.shape[0], -1))
                labels = Variable(labels)

            features = features.view(-1, 1, 28, 28)

            label_predict = ann(features)
            loss = criterion(label_predict, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_ce_iter += loss.item()
            loss_acc_iter += getHitCount(labels, label_predict)

        loss_ce_iter /= instance_count
        loss_acc_iter /= instance_count
        print('Epoch: {}, Cross Entropy Loss: {:.6f}, Classifier Accuracy: {:.6f}'
              .format(step+1, loss_ce_iter, loss_acc_iter))


def eval_model(ann, dl):
    criterion = nn.CrossEntropyLoss()
    instance_count = dl.dataset.__len__()
    loss_ce_iter, loss_acc_iter = .0, .0
    for features, labels in dl:
        if torch.cuda.is_available():
            features = Variable(features.view(features.shape[0], -1).cuda())
            labels = Variable(labels.cuda())

        else:
            features = Variable(features.view(features.shape[0], -1))
            labels = Variable(labels)
        features = features.view(-1, 1, 28, 28)

        label_predict = ann(features)
        loss = criterion(label_predict, labels)
        loss_ce_iter += loss.item()
        loss_acc_iter += getHitCount(labels, label_predict)

    loss_ce_iter /= instance_count
    loss_acc_iter /= instance_count
    print('Model performance in given model: Cross Entropy Loss: {:.6f}, Classifier Accuracy: {:.6f}'
          .format(loss_ce_iter, loss_acc_iter))


def main(load_flag=False):
    root_path = os.getcwd()
    clf_model = LeNetAE28()

    train_dl, test_dl = get_dl('mnist', root_path, True), get_dl('mnist', root_path, False)

    if load_flag:
        clf_model.load_state_dict(torch.load(root_path + '/../modeinfo/trained_model.pt'))
        clf_model.eval()
        eval_model(clf_model, test_dl)

    if torch.cuda.is_available():
        clf_model = clf_model.cuda()

    if not load_flag:
        train_model(clf_model, train_dl)
        eval_model(clf_model, test_dl)
    pass


if __name__ == "__main__":
    load_flag = False
    if len(sys.argv) > 1:
        load_flag = bool(sys.argv[1] == 'True')
    main(load_flag)





