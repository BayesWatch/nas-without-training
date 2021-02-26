import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
from statistics import mean
import time
from utils import add_dropout


parser = argparse.ArgumentParser(description='NAS Without Training')
parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results/ICML', type=str, help='folder to save results')
parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--kernel', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
parser.add_argument('--sigma', default=0.05, type=float, help='noise level if augtype is "gaussnoise"')
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--activations', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=500, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, ints = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), ints.detach()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, args)
os.makedirs(args.save_loc, exist_ok=True)


times     = []
chosen    = []
acc       = []
val_acc   = []
topscores = []
order_fn = np.nanargmax


if args.dataset == 'cifar10':
    acc_type = 'ori-test'
    val_acc_type = 'x-valid'
else:
    acc_type = 'x-test'
    val_acc_type = 'x-valid'



runs = trange(args.n_runs, desc='acc: ')
for N in runs:
    start = time.time()
    indices = np.random.randint(0,len(searchspace),args.n_samples)
    scores = []

    npstate = np.random.get_state()
    ranstate = random.getstate()
    torchstate = torch.random.get_rng_state()
    for arch in indices:
        try:
            uid = searchspace[arch]
            network = searchspace.get_network(uid)
            network.to(device)
            if args.dropout:
                add_dropout(network, args.sigma)
            if args.init != '':
                init_network(network, args.init)
            if 'hook_' in args.score:
                network.K = np.zeros((args.batch_size, args.batch_size))
                def counting_forward_hook(module, inp, out):
                    try:
                        if not module.visited_backwards:
                            return
                        if isinstance(inp, tuple):
                            inp = inp[0]
                        inp = inp.view(inp.size(0), -1)
                        x = (inp > 0).float()
                        K = x @ x.t()
                        K2 = (1.-x) @ (1.-x.t())
                        network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
                    except:
                        pass


                def counting_backward_hook(module, inp, out):
                    module.visited_backwards = True


                for name, module in network.named_modules():
                    if 'ReLU' in str(type(module)):
                        #hooks[name] = module.register_forward_hook(counting_hook)
                        module.register_forward_hook(counting_forward_hook)
                        module.register_backward_hook(counting_backward_hook)

            random.setstate(ranstate)
            np.random.set_state(npstate)
            torch.set_rng_state(torchstate)

            data_iterator = iter(train_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to(device)
            x, target = x.to(device), target.to(device)
            jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

            if args.kernel:
                s = get_score_func(args.score)(out, labels)
            elif 'hook_' in args.score:
                network(x2.to(device))
                s = get_score_func(args.score)(network.K, target)
            elif args.repeat < args.batch_size:
                s = get_score_func(args.score)(jacobs, labels, args.repeat)
            else:
                s = get_score_func(args.score)(jacobs, labels)
                
        except Exception as e:
            print(e)
            s = 0.
        
        scores.append(s)

    #print(len(scores))
    #print(scores)
    #print(order_fn(scores))

    

    best_arch = indices[order_fn(scores)]
    uid = searchspace[best_arch]
    topscores.append(scores[order_fn(scores)])
    chosen.append(best_arch)
    #acc.append(searchspace.get_accuracy(uid, acc_type, args.trainval))
    acc.append(searchspace.get_final_accuracy(uid, acc_type, False))

    if not args.dataset == 'cifar10' or args.trainval:
        val_acc.append(searchspace.get_final_accuracy(uid, val_acc_type, args.trainval))
    #    val_acc.append(info.get_metrics(dset, val_acc_type)['accuracy'])

    times.append(time.time()-start)
    runs.set_description(f"acc: {mean(acc):.2f}% time:{mean(times):.2f}")

print(f"Final mean test accuracy: {np.mean(acc)}")
#if len(val_acc) > 1:
#    print(f"Final mean validation accuracy: {np.mean(val_acc)}")

state = {'accs': acc,
         'chosen': chosen,
         'times': times,
         'topscores': topscores,
         }

dset = args.dataset if not (args.trainval and args.dataset == 'cifar10') else 'cifar10-valid'
fname = f"{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{dset}_{args.kernel}_{args.dropout}_{args.augtype}_{args.sigma}_{args.repeat}_{args.batch_size}_{args.n_runs}_{args.n_samples}_{args.seed}.t7"
torch.save(state, fname)
