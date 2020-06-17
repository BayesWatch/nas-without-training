import os
import argparse
import random
import numpy as np

import matplotlib.pyplot as plt
from datasets import get_datasets
from config_utils import load_config

from nas_201_api import NASBench201API as API
from models import get_cell_based_tiny_net
import torch
import torch.nn as nn


def get_batch_jacobian(net, data_loader, device):
    data_iterator = iter(data_loader)
    x, target = next(data_iterator)
    x = x.to(device)
    net.zero_grad()
    x.requires_grad_(True)
    _, y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()

def plot_hist(jacob, ax, colour):
    xx =  jacob.reshape(jacob.size(0), -1).cpu().numpy()
    corrs = np.corrcoef(xx)
    ax.hist(corrs.flatten(), bins=100, color=colour)

def decide_plot(acc, plt_cts, num_rows, boundaries=[60., 70., 80., 90.]):
    if acc < boundaries[0]:
        plt_col = 0
        accrange = f'< {boundaries[0]}%'
    elif acc < boundaries[1]:
        plt_col = 1
        accrange = f'[{boundaries[0]}% , {boundaries[1]}%)'
    elif acc < boundaries[2]:
        plt_col = 2
        accrange = f'[{boundaries[1]}% , {boundaries[2]}%)'
    elif acc < boundaries[3]:
        accrange = f'[{boundaries[2]}% , {boundaries[3]}%)'
        plt_col = 3
    else:
        accrange = f'>= {boundaries[3]}%'
        plt_col = 4

    can_plot = False
    plt_row = 0
    if plt_cts[plt_col] < num_rows:
        can_plot = True
        plt_row = plt_cts[plt_col]
        plt_cts[plt_col] += 1

    return can_plot, plt_row, plt_col, accrange



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot histograms of correlation matrix')
    parser.add_argument('--data_loc', default='../datasets/cifar/', type=str, help='dataset folder')
    parser.add_argument('--api_loc', default='NAS-Bench-201-v1_1-096897.pth',
                    type=str, help='path to API')
    parser.add_argument('--arch_start', default=0, type=int)
    parser.add_argument('--arch_end', default=15625, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--GPU', default='0', type=str)
    parser.add_argument('--batch_size', default=256, type=int)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ARCH_START = args.arch_start
    ARCH_END = args.arch_end

    criterion = nn.CrossEntropyLoss()
    train_data, valid_data, xshape, class_num = get_datasets('cifar10', args.data_loc, 0)

    cifar_split = load_config('config_utils/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       num_workers=0, pin_memory=True, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

    scores = []
    accs = []

    plot_shape = (25, 5)
    num_plots = plot_shape[0]*plot_shape[1]
    fig, axes = plt.subplots(*plot_shape, sharex=True, figsize=(9, 9) )
    plt_cts = [0 for i in range(plot_shape[1])]

    api = API(args.api_loc)

    archs = list(range(ARCH_START, ARCH_END))
    colours = ['#811F41', '#A92941', '#D15141', '#EF7941', '#F99C4B']

    strs = []
    random.shuffle(archs)
    for arch in archs:
        try:
            config = api.get_net_config(arch, 'cifar10')
            archinfo = api.query_meta_info_by_index(arch)
            acc = archinfo.get_metrics('cifar10-valid', 'x-valid')['accuracy']

            network = get_cell_based_tiny_net(config)
            network = network.to(device)
            jacobs, labels = get_batch_jacobian(network, train_loader, device)

            boundaries = [60., 70., 80., 90.]
            can_plt, row, col, accrange = decide_plot(acc, plt_cts, plot_shape[0], boundaries)
            if not can_plt:
                continue
            axes[row, col].axis('off')

            plot_hist(jacobs, axes[row, col], colours[col])
            if row == 0:
                axes[row, col].set_title(f'{accrange}')

            if row + 1 == plot_shape[0]:
                axes[row, col].axis('on')
                plt.setp(axes[row, col].get_xticklabels(), fontsize=12)
                axes[row, col].spines["top"].set_visible(False)
                axes[row, col].spines["right"].set_visible(False)
                axes[row, col].spines["left"].set_visible(False)
                axes[row, col].set_yticks([])

            if sum(plt_cts) == num_plots:
                plt.tight_layout()
                plt.savefig(f'results/histograms_cifar10val_batch{args.batch_size}.png')
                plt.show()
                break
        except Exception as e:
            plt_cts[col] -= 1
            continue
