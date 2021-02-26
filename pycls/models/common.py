#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from pycls.core.config import cfg


def Preprocess(x):
    if cfg.TASK == 'jig':
        assert len(x.shape) == 5, 'Wrong tensor dimension for jigsaw'
        assert x.shape[1] == cfg.JIGSAW_GRID ** 2, 'Wrong grid for jigsaw'
        x = x.view([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
    return x


class Classifier(nn.Module):
    def __init__(self, channels, num_classes):
        super(Classifier, self).__init__()
        if cfg.TASK == 'jig':
            self.jig_sq = cfg.JIGSAW_GRID ** 2
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(channels * self.jig_sq, num_classes)
        elif cfg.TASK == 'col':
            self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        elif cfg.TASK == 'seg':
            self.classifier = ASPP(channels, cfg.MODEL.ASPP_CHANNELS, num_classes, cfg.MODEL.ASPP_RATES)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x, shape):
        if cfg.TASK == 'jig':
            x = self.pooling(x)
            x = x.view([x.shape[0] // self.jig_sq, x.shape[1] * self.jig_sq, x.shape[2], x.shape[3]])
            x = self.classifier(x.view(x.size(0), -1))
        elif cfg.TASK in ['col', 'seg']:
            x = self.classifier(x)
            x = nn.Upsample(shape, mode='bilinear', align_corners=True)(x)
        else:
            x = self.pooling(x)
            x = self.classifier(x.view(x.size(0), -1))
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, rates):
        super(ASPP, self).__init__()
        assert len(rates) in [1, 3]
        self.rates = rates
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, dilation=rates[0],
                padding=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if len(self.rates) == 3:
            self.aspp3 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, dilation=rates[1],
                    padding=rates[1], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.aspp4 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, dilation=rates[2],
                    padding=rates[2], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.aspp5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                align_corners=True)(x5)
        if len(self.rates) == 3:
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x = torch.cat((x1, x2, x3, x4, x5), 1)
        else:
            x = torch.cat((x1, x2, x5), 1)
        x = self.classifier(x)
        return x
