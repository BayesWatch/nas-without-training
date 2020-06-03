import numpy as np
import argparse
import os
import random
import pandas as pd
from collections import OrderedDict

import tabulate
parser = argparse.ArgumentParser(description='Produce tables')
parser.add_argument('--data_loc', default='../datasets/cifar/', type=str, help='dataset folder')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')

parser.add_argument('--n_samples', default=100, type=int, help='how many samples to take')
parser.add_argument('--n_runs', default=500, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

from statistics import mean, median, stdev as std

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

df = []

datasets = OrderedDict()

datasets['CIFAR-10 (val)'] = ('cifar10-valid', 'x-valid', True)
datasets['CIFAR-10 (test)'] = ('cifar10', 'ori-test', False)

### CIFAR-100
datasets['CIFAR-100 (val)'] = ('cifar100', 'x-valid', False)
datasets['CIFAR-100 (test)'] = ('cifar100', 'x-test', False)

datasets['ImageNet16-120 (val)'] = ('ImageNet16-120', 'x-valid', False)
datasets['ImageNet16-120 (test)'] = ('ImageNet16-120', 'x-test', False)


dataset_top1s = OrderedDict()
dataset_top1s['Method'] = f"Ours (N={args.n_samples})"
dataset_top1s['Search time (s)'] = np.nan

time = 0.

for dataset, params in datasets.items():
    top1s = []

    dset =  params[0]
    acc_type = 'accs' if 'test' in params[1] else 'val_accs'
    filename = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"

    full_scores = torch.load(filename)
    if dataset == 'CIFAR-10 (test)':
        time = median(full_scores['times'])
        dataset_top1s['Search time (s)'] = time
    accs = []
    for n in range(args.n_runs):
        acc = full_scores[acc_type][n]
        accs.append(acc)
    dataset_top1s[dataset] = accs

df = pd.DataFrame(dataset_top1s)

df['CIFAR-10 (val)']  = f"{mean(df['CIFAR-10 (val)']):.2f} +- {std(df['CIFAR-10 (val)']):.2f}"
df['CIFAR-10 (test)']  = f"{mean(df['CIFAR-10 (test)']):.2f} +- {std(df['CIFAR-10 (test)']):.2f}"

df['CIFAR-100 (val)'] = f"{mean(df['CIFAR-100 (val)']):.2f} +- {std(df['CIFAR-100 (val)']):.2f}"
df['CIFAR-100 (test)'] = f"{mean(df['CIFAR-100 (test)']):.2f} +- {std(df['CIFAR-100 (test)']):.2f}"

df['ImageNet16-120 (val)']  = f"{mean(df['ImageNet16-120 (val)']):.2f} +- {std(df['ImageNet16-120 (val)']):.2f}"
df['ImageNet16-120 (test)'] = f"{mean(df['ImageNet16-120 (test)']):.2f} +- {std(df['ImageNet16-120 (test)']):.2f}"

df = df.round(2)
df = df.iloc[:1]
print(tabulate.tabulate(df.values,df.columns, tablefmt="pipe"))
