import os
import time
import argparse
import random
import numpy as np
from tqdm import trange
from statistics import mean

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../datasets/NAS-Bench-201-v1_1-096897.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch.optim as optim

from models import get_cell_based_tiny_net

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import torchvision.transforms as transforms
from datasets import get_datasets
from config_utils import load_config
from nas_201_api import NASBench201API as API

def get_batch_jacobian(net, x, target, to, device, args=None):
    net.zero_grad()

    x.requires_grad_(True)

    _, y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
THE_START = time.time()
api = API(args.api_loc)

os.makedirs(args.save_loc, exist_ok=True)

train_data, valid_data, xshape, class_num = get_datasets(args.dataset, args.data_loc, cutout=0)

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'

else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'

if args.trainval:
    cifar_split = load_config('config_utils/cifar-split.txt', None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))

else:
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True)

times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []

dset = args.dataset if not args.trainval else 'cifar10-valid'

order_fn = np.nanargmax

runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    indices = np.random.randint(0,15625,args.n_samples)
    scores = []

    for arch in indices:

        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x, target = x.to(device), target.to(device)

        config = api.get_net_config(arch, args.dataset)
        config['num_classes'] = 1

        network = get_cell_based_tiny_net(config)  # create the network from configuration
        network = network.to(device)

        jacobs, labels= get_batch_jacobian(network, x, target, 1, device, args)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        try:
            s = eval_score(jacobs, labels)
        except Exception as e:
            print(e)
            s = np.nan

        scores.append(s)

    best_arch = indices[order_fn(scores)]
    info      = api.query_by_index(best_arch)
    topscores.append(scores[order_fn(scores)])
    chosen.append(best_arch)
    acc.append(info.get_metrics(dset, acc_type)['accuracy'])

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc if not args.trainval else val_acc):.2f}%")

print(f"Final mean test accuracy: {np.mean(acc)}")
if len(val_acc) > 1:
    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'accs': acc,
         'val_accs': val_acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not args.trainval else 'cifar10-valid'
fname = f"{args.save_loc}/{dset}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
