import torch
from copy import deepcopy


def copy_conv(src_model, tgt_model):
    tgt_model.feature_extractor = deepcopy(src_model.feature_extractor)
    pass

