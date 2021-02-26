#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet
from pycls.models.regnet import RegNet
from pycls.models.resnet import ResNet
from pycls.models.nas.nas import NAS
from pycls.models.nas.nas_search import NAS_Search
from pycls.models.nas_bench.model_builder import NAS_Bench


class LabelSmoothedCrossEntropyLoss(torch.nn.Module):
    """CrossEntropyLoss with label smoothing."""
    def __init__(self):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.eps = cfg.MODEL.LABEL_SMOOTHING_EPS
        self.num_classes = cfg.MODEL.NUM_CLASSES

    def forward(self, logits, target):
        pred = logits.log_softmax(dim=-1)
        with torch.no_grad():
            target_dist = torch.ones_like(pred) * self.eps / (self.num_classes - 1)
            target_dist.scatter_(-1, target.unsqueeze(-1), 1 - self.eps)
        return (-target_dist * pred).sum(dim=-1).mean()


# Supported models
_models = {
    "anynet": AnyNet,
    "effnet": EffNet,
    "resnet": ResNet,
    "regnet": RegNet,
    "nas": NAS,
    "nas_search": NAS_Search,
    "nas_bench": NAS_Bench,
}

# Supported loss functions
_loss_funs = {
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "label_smoothed_cross_entropy": LabelSmoothedCrossEntropyLoss,
}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun():
    """Build the loss function."""
    if cfg.TASK == "seg":
        return get_loss_fun()(ignore_index=255)
    else:
        return get_loss_fun()()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
