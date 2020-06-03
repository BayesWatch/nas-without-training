#!/bin/bash

python search.py --dataset cifar10 --data_loc '../datasets/cifar10'            --n_runs $1 --n_samples 10
python search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 10
python search.py --dataset cifar100 --data_loc '../datasets/cifar100'          --n_runs $1 --n_samples 10
python search.py --dataset ImageNet16-120 --data_loc '../datasets/ImageNet16'  --n_runs $1 --n_samples 10

python search.py --dataset cifar10 --data_loc '../datasets/cifar10'            --n_runs $1 --n_samples 100
python search.py --dataset cifar10 --trainval --data_loc '../datasets/cifar10' --n_runs $1 --n_samples 100
python search.py --dataset cifar100 --data_loc '../datasets/cifar100'          --n_runs $1 --n_samples 100
python search.py --dataset ImageNet16-120 --data_loc '../datasets/ImageNet16'  --n_runs $1 --n_samples 100

python process_results.py --n_runs $1
