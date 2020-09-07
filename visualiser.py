import re
from graphviz import Digraph
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser(description='Fast cell visualisation')
parser.add_argument('--arch', default=1, type=int)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

def set_none(bit):
    print(bit)
    tmp = bit.split('~')
    tmp[0] = 'none'
    print('~'.join(tmp))
    return '~'.join(tmp)

def remove_pointless_ops(archstr):
    old = None
    new = archstr
    while old != new:
        old = new
        bits = old.strip('|').split('|')
        if 'none~' in bits[0]: # node 1 has no connections to it
            bits[3] =  set_none(bits[3]) # node 1 -> 2 now none
            bits[6] =  set_none(bits[6]) # node 1 -> 3 now none
        if 'none~' in bits[2] and 'none~' in bits[3]: # node 2 has no connections to it
            bits[7] =  set_none(bits[7]) # node 2 -> 3 now none
        if 'none~' in bits[7]: # doesn't matter what comes through node 2
            bits[2] =  set_none(bits[2]) # node 0 -> 2 now none
            bits[3] =  set_none(bits[3]) # node 1 -> 2 now none
        if 'none~' in bits[6] and 'none~' in bits[7]: # doesn't matter what comes through node 1
            bits[0] =  set_none(bits[0]) # node 0 -> 1 now none
        new = '|'.join(bits)
    print(new)
    return new


df = pd.read_pickle('results/arch_score_acc.pd')

nodestr = df.iloc[args.arch]['cellstr']
nodestr = nodestr[1:-1] # remove leading and trailing bars |

nodestr = remove_pointless_ops(nodestr)
nodes = nodestr.split("|+|")

dot = Digraph(
  format='pdf',
  edge_attr=dict(fontsize='12'),
  node_attr=dict(fixedsize='true',shape="circle", height='0.5', width='0.5'),
  engine='dot')

dot.body.extend(['rankdir=LR'])

OPS = ['conv_3x3','avg_pool_3x3','skip_connect','conv_1x1','none']

dot.node('0', 'in')

## ops are separated by bars (|) so
for i, node in enumerate(nodes):

    # if node 3 then label as output
    if (i+1) == 3:
        dot.node(str(i+1), 'out')
    else:
        dot.node(str(i+1))

    for op_str in node.split('|'):
        op_name = [o for o in OPS if o in op_str][0]
        if op_name == 'none':
            break
        connect = re.findall('~[0-9]', op_str)[0]
        connect = connect[1:]
        dot.edge(connect,str(i+1), label=op_name)

dot.render( view=True)


if args.save:
    dot.render(f'outputs/{args.arch}.gv')
