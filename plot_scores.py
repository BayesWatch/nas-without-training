import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib
matplotlib.use('Agg')
from decimal import Decimal
from scipy.special import logit, expit
from scipy import stats
import seaborn as sns

'''
font = {
        'size'   : 18}

matplotlib.rc('font', **font)
'''
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--init', default='', type=str)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()

print(f'{args.batch_size}')
random.seed(args.seed)
np.random.seed(args.seed)

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{args.dataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}.npy'
accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{args.dataset}_{args.trainval}.npy'

from matplotlib.colors import hsv_to_rgb
print(filename)
scores = np.load(filename)
accs = np.load(accfilename)

def make_colours_by_hue(h, v=1.):
    return [hsv_to_rgb((h1 if h1 < 1. else h1-1., s, v)) for h1, s,v in zip(np.linspace(h, h+0.05, 5), np.linspace(1., .6, 5), np.linspace(0.1, 1., 5))]
print(f'NETWORK accuracy with highest score {accs[np.argmax(scores)]}')

make_colours = lambda cols: [mp.colors.to_rgba(c) for c in cols]
oranges = make_colours(['#811F41', '#A92941', '#D15141', '#EF7941', '#F99C4B'])
blues = make_colours(['#190C30', '#241147', '#34208C', '#4882FA', '#81BAFC'])
print(blues)
print(make_colours_by_hue(0.9))
if args.nasspace == 'nasbench101':
    #colours = blues
    colours = make_colours_by_hue(0.9)
elif 'darts' in args.nasspace:
    #colours = sns.color_palette("BuGn_r", n_colors=5)
    colours = make_colours_by_hue(0.0)
elif 'pnas' in args.nasspace:
    #colours = sns.color_palette("PuRd", n_colors=5)
    colours = make_colours_by_hue(0.1)
elif args.nasspace == 'nasbench201':
    #colours = oranges
    colours = make_colours_by_hue(0.3)
elif 'enas' in args.nasspace:
    #colours = oranges
    colours = make_colours_by_hue(0.4)
elif 'resnet' in args.nasspace:
    #colours = sns.color_palette("viridis_r", n_colors=5)
    colours = make_colours_by_hue(0.5)
elif 'amoeba' in args.nasspace:
    #colours = sns.color_palette("viridis_r", n_colors=5)
    colours = make_colours_by_hue(0.6)
elif 'nasnet' in args.nasspace:
    #colours = sns.color_palette("viridis_r", n_colors=5)
    colours = make_colours_by_hue(0.7)
elif 'resnext-b' in args.nasspace:
    #colours = sns.color_palette("viridis_r", n_colors=5)
    colours = make_colours_by_hue(0.8)
else:
    from zlib import crc32

    def bytes_to_float(b):
        return float(crc32(b) & 0xffffffff) / 2**32
    def str_to_float(s, encoding="utf-8"):
        return bytes_to_float(s.encode(encoding))
    #colours = sns.color_palette("Purples_r", n_colors=5)
    colours = make_colours_by_hue(str_to_float(args.nasspace))

def make_colordict(colours, points):
    cdict = {'red': [[pt, colour[0], colour[0]] for pt, colour in zip(points, colours)],
             'green':[[pt, colour[1], colour[1]] for pt, colour in zip(points, colours)],
             'blue':[[pt, colour[2], colour[2]] for pt, colour in zip(points, colours)]}
    return cdict

def make_colormap(dataset, space, colours):
    if dataset == 'cifar10' and 'resn' in space:
        points = [0., 0.85, 0.9, 0.95, 1.0, 1.0]
        colours = [colours[0]] + colours
    elif dataset == 'cifar10' and 'nds_darts' in space:
        points = [0., 0.8, 0.85, 0.9, 0.95, 1.0]
        colours = [colours[0]] + colours
    elif dataset == 'cifar10' and 'pnas' in space:
        points = [0., 0.875, 0.9, 0.925, 0.95, 1.0]
        colours = [colours[0]] + colours
    elif dataset == 'cifar10':
        points = [0., 0.6, 0.7, 0.8, 0.9, 1.0]
        colours = [colours[0]] + colours
        #cdict = {'red': [[0., colours[0][0], colours[0][0]]] + [[0.1*i + 0.6, colours[i][0], colours[i][0]] for i in range(len(colours))],
        #         'green':[[0., colours[0][1], colours[0][1]]] +  [[0.1*i + 0.6, colours[i][1], colours[i][1]] for i in range(len(colours))],
        #         'blue':[[0., colours[0][2], colours[0][2]]] +  [[0.1*i + 0.6, colours[i][2], colours[i][2]] for i in range(len(colours))]}
    elif dataset == 'cifar100':
        points = [0., 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        colours = [colours[0]] + colours + [colours[-1]]
    
        #cdict = {'red': [[0., colours[0][0], colours[0][0]]] + [[0.1*i + 0.3, colours[i][0], colours[i][0]]   for i in range(len(colours))]  + [[1., colours[-1][0], colours[-1][0]]] ,
        #         'green':[[0., colours[0][1], colours[0][1]]] +  [[0.1*i + 0.3, colours[i][1], colours[i][1]]  for i in range(len(colours))] + [[1., colours[-1][1], colours[-1][1]]] ,
        #         'blue':[[0., colours[0][2], colours[0][2]]] +  [[0.1*i + 0.3, colours[i][2], colours[i][2]]  for i in range(len(colours))]  + [[1., colours[-1][2], colours[-1][2]]] }
    else:
        points = [0., 0.1, 0.2, 0.3, 0.4, 1.0]
        colours = colours + [colours[-1]]
    
        #cdict = {'red': [[0.1*i, colours[i][0], colours[i][0]]    for i in range(len(colours))] + [[1., colours[-1][0], colours[-1][0]]] ,
        #         'green': [[0.1*i, colours[i][1], colours[i][1]]  for i in range(len(colours))] + [[1., colours[-1][1], colours[-1][1]]] ,
        #         'blue': [[0.1*i, colours[i][2], colours[i][2]]   for i in range(len(colours))] + [[1., colours[-1][2], colours[-1][2]]] }
    
    cdict = make_colordict(colours, points)    
    return cdict
cdict = make_colormap(args.dataset, args.nasspace, colours)
newcmp = mp.colors.LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)

if args.nasspace == 'nasbench101':
    accs = accs[:10000]
    scores = scores[:10000]
    inds = accs > 0.5
    accs = accs[inds]
    scores = scores[inds]
    print(accs.shape)
elif args.nasspace == 'nds_amoeba' or args.nasspace == 'nds_darts_fix-w-d':
    print(accs.shape)
    inds = accs > 15.
    accs = accs[inds]
    scores = scores[inds]
    print(accs.shape)
elif args.nasspace == 'nds_darts':
    inds = accs > 15.
    from nasspace import get_search_space
    searchspace = get_search_space(args)
    accs = accs[inds]
    scores = scores[inds]
    print(accs.shape)
else:
    print(accs.shape)
    inds = accs > 15.
    accs = accs[inds]
    scores = scores[inds]
    print(accs.shape)

inds = scores == 0.
accs = accs[~inds]
scores = scores[~inds]



if accs.size > 1000:
    inds = np.random.choice(accs.size, 1000, replace=False)
    accs = accs[inds]
    scores = scores[inds]

inds = np.isnan(scores)
accs = accs[~inds]
scores = scores[~inds]

tau, p = stats.kendalltau(accs, scores)

if args.nasspace == 'nasbench101':
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
else:
    fig, ax = plt.subplots(1, 1, figsize=(5,5))

def scale(x):
    return 2.**(10*x) - 1.

if args.score == 'svd':
    score_scale = lambda x: 10.0**x 
else:
    score_scale = lambda x: x

if args.nasspace == 'nonetwork':
    ax.scatter(scale(accs/100.), score_scale(scores), c=newcmp(accs/100., depths))
else:
    ax.scatter(scale(accs/100. if args.nasspace == 'nasbench201' or 'nds' in args.nasspace else accs), score_scale(scores), c=newcmp(accs/100. if args.nasspace == 'nasbench201' or 'nds' in args.nasspace else accs))


if args.dataset == 'cifar100':
    ax.set_xticks([scale(float(a)/100.) for a in [40, 60, 70]])
    ax.set_xticklabels([f'{a}' for a in [40, 60, 70]])
elif args.dataset == 'imagenette2':
    ax.set_xticks([scale(float(a)/100.) for a in [40, 50, 60, 70]])
    ax.set_xticklabels([f'{a}' for a in [40, 50, 60, 70]])
elif args.dataset == 'ImageNet16-120':
    ax.set_xticks([scale(float(a)/100.) for a in [20, 30, 40, 45]])
    ax.set_xticklabels([f'{a}' for a in [20, 30, 40, 45]])
elif args.nasspace == 'nasbench101' and args.dataset == 'cifar10':
    ax.set_xticks([scale(float(a)/100.) for a in [50, 80, 90, 95]])
    ax.set_xticklabels([f'{a}' for a in [50, 80, 90, 95]])
elif args.nasspace == 'nasbench201' and args.dataset == 'cifar10' and args.score == 'svd':
    ax.set_xticks([scale(float(a)/100.) for a in [50, 80, 90, 95]])
    ax.set_xticklabels([f'{a}' for a in [50, 80, 90, 95]])
elif 'nds_resne' in args.nasspace and args.dataset == 'cifar10':
    ax.set_xticks([scale(float(a)/100.) for a in [85, 88, 91, 94]])
    ax.set_xticklabels([f'{a}' for a in [85, 88, 91, 94]])
elif args.nasspace == 'nds_darts' and args.dataset == 'cifar10':
    ax.set_xticks([scale(float(a)/100.) for a in [80, 85, 90, 95]])
    ax.set_xticklabels([f'{a}' for a in [80, 85, 90, 95]])
elif args.nasspace == 'nds_pnas' and args.dataset == 'cifar10':
    ax.set_xticks([scale(float(a)/100.) for a in [90., 91.5, 93, 94.5]])
    ax.set_xticklabels([f'{a}' for a in [90., 91.5, 93, 94.5]])
else:
    ax.set_xticks([scale(float(a)/100.) for a in [50, 80, 90]])
    ax.set_xticklabels([f'{a}' for a in [50, 80, 90]])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

nasspacenames = {
    'nds_resnext-a_in': 'NDS-ResNeXt-A(ImageNet)',
    'nds_resnext-b_in': 'NDS-ResNeXt-B(ImageNet)',
    'nds_resnext-a': 'NDS-ResNeXt-A(CIFAR10)',
    'nds_resnext-b': 'NDS-ResNeXt-B(CIFAR10)',
    'nds_nasnet': 'NDS-NASNet(CIFAR10)',
    'nds_nasnet_in': 'NDS-NASNet(ImageNet)',
    'nds_enas': 'NDS-ENAS(CIFAR10)',
    'nds_enas_in': 'NDS-ENAS(ImageNet)',
    'nds_amoeba': 'NDS-Amoeba(CIFAR10)',
    'nds_amoeba_in': 'NDS-Amoeba(ImageNet)',
    'nds_resnet': 'NDS-ResNet(CIFAR10)',
    'nds_pnas': 'NDS-PNAS(CIFAR10)',
    'nds_pnas_in': 'NDS-PNAS(ImageNet)',
    'nds_darts': 'NDS-DARTS(CIFAR10)',
    'nds_darts_in': 'NDS-DARTS(ImageNet)',
    'nds_darts_fix-w-d': 'NDS-DARTS fixed width/depth (CIFAR10)',
    'nds_darts_in_fix-w-d': 'NDS-DARTS fixed width/depth (ImageNet)',
    'nds_darts_in': 'NDS-DARTS(ImageNet)',
    'nasbench101': 'NAS-Bench-101',
    'nasbench201': 'NAS-Bench-201'
}

ax.set_ylabel('Score')
ax.set_xlabel(f'{"Test" if not args.trainval else "Validation"} accuracy')
ax.set_title(f'{nasspacenames[args.nasspace]} {args.dataset} \n $\\tau=${tau:.3f}')

filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{args.dataset}{"_" + args.init + "_" if args.init != "" else args.init}{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
print(filename)
plt.tight_layout()
plt.savefig(filename + '.pdf')
plt.savefig(filename + '.png')

plt.show()
