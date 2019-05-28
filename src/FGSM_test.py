import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.datasets import MNIST

import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from Model import LeNetAE28, IRNet
from data_loader import get_dl, get_cmnist_dl
from torch.utils.data import DataLoader
from loss import ContrastiveLoss
from auxiliary import copy_conv
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
    root_path = os.getcwd()
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

    torch.save(ann.state_dict(), root_path + '/../modeinfo/trained_model.pt')


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


def train_siamese(ann, dl):
    criterion = ContrastiveLoss(margin=.7)
    optim = torch.optim.Adam(ann.parameters(), lr=1e-4)
    for (features_x, features_y), labels in dl:
        if torch.cuda.is_available():
            features_x = Variable(features_x.view(features_x.shape[0], -1).cuda())
            features_y = Variable(features_y.view(features_y.shape[0], -1).cuda())
            labels = Variable(labels.cuda())

        else:
            features_x = Variable(features_x.view(features_x.shape[0], -1))
            features_y = Variable(features_y.view(features_y.shape[0], -1))
            labels = Variable(labels)

        mac_x, mac_y = ann((features_x.view(-1, 1, 28, 28), features_y.view(-1, 1, 28, 28)))
        loss = criterion(mac_x, mac_y, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()


def aux_clip(d, min_value, max_value):
    idx = d < min_value
    d[idx] = min_value
    idx = d > max_value
    d[idx] = max_value
    return d


def fgsm_attack(ann, dl, epsilon):
    total = 0
    hit, hit_under_attack = 0, 0
    instance_count = dl.dataset.__len__()
    criterion = nn.CrossEntropyLoss()
    for i, (features, labels) in enumerate(dl):
        total += features.shape[0]
        if torch.cuda.is_available():
            features = Variable(features.view(features.shape[0], -1).view(-1, 1, 28, 28).cuda(), requires_grad=True)
            labels = Variable(labels.cuda())

        else:
            features = Variable(features.view(features.shape[0], -1).view(-1, 1, 28, 28), requires_grad=True)
            labels = Variable(labels)

        label_predict = ann(features)
        hit += getHitCount(labels, label_predict)
        ann.zero_grad()
        loss = criterion(label_predict, labels)
        loss.backward()

        fake_features = aux_clip((features + epsilon * torch.sign(features.grad.data)), 0, 1)
        if torch.cuda.is_available():
            fake_features = Variable(fake_features.cuda())
        else:
            fake_features = Variable(fake_features)

        label_predict_2 = ann(fake_features)
        hit_under_attack += getHitCount(labels, label_predict_2)
        print('Step: {}'.format(i))

    hit /= instance_count
    hit_under_attack /= instance_count
    print('Acc before attack: {:.6f}, Acc after attack: {:.6f}'.format(hit, hit_under_attack))


def main(load_flag=False):
    root_path = os.getcwd()
    clf_model = LeNetAE28()

    train_dl, test_dl = get_dl('mnist', root_path, True), get_dl('mnist', root_path, False)

    if load_flag:
        clf_model.load_state_dict(torch.load(root_path + '/../modeinfo/trained_model.pt'))
        clf_model.eval()

    if torch.cuda.is_available():
        clf_model = clf_model.cuda()
        if load_flag:
            eval_model(clf_model, test_dl)

    if not load_flag:
        train_model(clf_model, train_dl)
        eval_model(clf_model, test_dl)

    print('FGSM Attack Start')
    fgsm_attack(clf_model, test_dl, epsilon=.06)

    print("Fine tuning the network via siamese architecture")
    imgRetrievalNet = IRNet()
    copy_conv(clf_model, imgRetrievalNet)

    CMNISTLoader = get_cmnist_dl(root_path, True)

    if torch.cuda.is_available():
        imgRetrievalNet = imgRetrievalNet.cuda()

    train_siamese(imgRetrievalNet, CMNISTLoader)

    pass


if __name__ == "__main__":
    load_flag = False
    if len(sys.argv) > 1:
        load_flag = bool(sys.argv[1] == 'True')
    main(load_flag)





