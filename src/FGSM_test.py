import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision.datasets import MNIST

import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from Model import LeNetAE28, IRNet
from data_loader import get_dl, get_ccifar_dl, get_dl_w_sampler
from torch.utils.data import DataLoader
from loss import ContrastiveLoss
from auxiliary import copy_conv
import heapq
import numpy as np
import sys
import os

trans = T.Compose([T.ToTensor()])
reverse_trans = lambda x: np.asarray(T.ToPILImage()(x))
_GLOBAL_AUX_FEATURE = None
_GLOBAL_AUX_LABEL = None

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
    for i in range(50):
        total_loss = .0
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
            loss.sum().backward()
            optim.step()

            total_loss += loss.sum().item()
        print("Epoch: {}, ContrastiveLoss: {:.6f}".format(i+1, total_loss))


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

    hit /= instance_count
    hit_under_attack /= instance_count
    print('Acc before attack: {:.6f}, Acc after attack: {:.6f}'.format(hit, hit_under_attack))


def fgsm_attack_retrieval(f_nn, r_nn, src_dl, test_dl, epsilon):
    top5, top10 = .0, .0
    top5_ua, top10_ua = .0, .0
    criterion = nn.CrossEntropyLoss()
    instance_count = test_dl.dataset.__len__()
    print('Number of instance need to be examed : {}'.format(instance_count))
    for i, (features, labels) in enumerate(test_dl):
        if torch.cuda.is_available():
            features = Variable(features.view(features.shape[0], -1).view(-1, 1, 28, 28).cuda(), requires_grad=True)
            labels = Variable(labels.cuda())

        else:
            features = Variable(features.view(features.shape[0], -1).view(-1, 1, 28, 28), requires_grad=True)
            labels = Variable(labels)

        label_predict = f_nn(features)
        f_nn.zero_grad()
        loss = criterion(label_predict, labels)
        loss.backward()

        fake_features = aux_clip((features + epsilon * torch.sign(features.grad.data)), 0, 1)
        if torch.cuda.is_available():
            fake_features = Variable(fake_features.cuda())
        else:
            fake_features = Variable(fake_features)

        real_ret_list = get_top_list(r_nn, features, src_dl)
        fake_ret_list = get_top_list(r_nn, fake_features, src_dl)
        tmp1, tmp2 = get_retrieval_hit(labels.item(), real_ret_list)
        top5 += tmp1
        top10 += tmp2
        print('Instance: {}, Top5 Acc: {:.6f}, Top10 Acc: {:.6f}'.format(i, tmp1, tmp2))

        tmp1, tmp2 = get_retrieval_hit(labels.item(), fake_ret_list)
        top5_ua += tmp1
        top10_ua += tmp2
        print('Instance:{}, Top5_UA Acc: {:.6f}, Top10_UA Acc: {:.6f}'.format(i, tmp1, tmp2))
    top5 /= 5 * instance_count
    top5_ua /= 5 * instance_count
    top10 /= 10 * instance_count
    top10_ua /= 10 * instance_count
    print("Top5 Accuracy: {:.6f}, Top10 Accuracy: {:.6f}, Top5 Accuracy UA: {:.6f}, Top10 Accuracy UA: {:.6F}".format(
        top5, top10, top5_ua, top10_ua
    ))


def get_feature_data_set(ann, dl):
    res_data = []
    res_label = []
    for f, l in dl:
        if torch.cuda.is_available():
            f = Variable(f.view(f.shape[0], -1).cuda())
            l = Variable(l.cuda())
        else:
            f = Variable(f.view(f.shape[0], -1))
            l = Variable(l)

        res = ann(f.view(-1, 1, 28, 28))
        res_data.append(res)
        res_label.append(l)
    return res_data, res_label


def get_top_list(rnn, input_instance, corpus_dl):
    aux_list = []
    c_mac = rnn(input_instance)
    global _GLOBAL_AUX_FEATURE, _GLOBAL_AUX_LABEL
    if _GLOBAL_AUX_FEATURE is None:
        _GLOBAL_AUX_FEATURE, _GLOBAL_AUX_LABEL = get_feature_data_set(rnn, corpus_dl)
    for i in range(len(_GLOBAL_AUX_FEATURE)):
        f, l = _GLOBAL_AUX_FEATURE[i], _GLOBAL_AUX_LABEL[i]
        dis = F.pairwise_distance(c_mac, f)

        if len(aux_list) < 20:
            heapq.heappush(aux_list, (-dis.item(), l.item()))
        else:
            heapq.heappushpop(aux_list, (-dis.item(), l.item()))
    return heapq.nlargest(10, aux_list)


def get_retrieval_hit(tgt_label, ret_list):
    t1, t2 = .0, .0
    for i in range(10):
        if tgt_label == ret_list[i][1]:
            t1 += 1. if i < 5 else .0
            t2 += 1.
    return t1, t2



def main(load_flag=False):
    root_path = os.getcwd()
    clf_model = LeNetAE28()

    # train_dl, test_dl = get_dl('mnist', root_path, True), get_dl('mnist', root_path, False)
    train_dl, test_dl = get_dl('cifar', root_path, True), get_dl('cifar', root_path, False)

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
    fgsm_attack(clf_model, test_dl, epsilon=.3)

    print("Fine tuning the network via siamese architecture")
    imgRetrievalNet = IRNet()
    copy_conv(clf_model, imgRetrievalNet)

    # CMNISTLoader = get_cmnist_dl(root_path, True)
    CCIFARLoader = get_ccifar_dl(root_path, True)

    if torch.cuda.is_available():
        imgRetrievalNet = imgRetrievalNet.cuda()

    train_siamese(imgRetrievalNet, CCIFARLoader)

    print('FGSM Attack -- Retrieval Model')
    train_dl, test_dl = get_dl('cifar', root_path, True, batch_size=1), get_dl_w_sampler('cifar', root_path, False, batch_size=1)
    fgsm_attack_retrieval(clf_model, imgRetrievalNet, train_dl, test_dl, .3)

    pass


if __name__ == "__main__":
    load_flag = False
    if len(sys.argv) > 1:
        load_flag = bool(sys.argv[1] == 'True')
    main(load_flag)





