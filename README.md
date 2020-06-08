# Neural Architecture Search Without Training

This repository contains code for replicating our paper on NAS without training. 

## Setup 

1. Download the [datasets](https://drive.google.com/drive/folders/1L0Lzq8rWpZLPfiQGd6QR8q5xLV88emU7).
2. Download [NAS-Bench-201](https://drive.google.com/file/d/1OOfVPpt-lA4u2HJrXbgrRd42IbfvJMyE/view).
3. Install the requirements in a conda environment with `conda env create -f environment.yml`.

We also refer the reader to instructions in the official [NASBench-201 README](https://github.com/D-X-Y/NAS-Bench-201).

## Reproducing our results 

To reproduce our results:

```
conda activate nas-wot
./reproduce.sh 3 # average accuracy over 3 runs
./reproduce.sh 500 # average accuracy over 500 runs (this will take longer)
```

Each command will finish by calling `process_results.py`, which will print a table. `./reproduce.sh 3` should print the following table:

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   |
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=10)  |           1.73435 | 89.25 +- 0.08    | 92.21 +- 0.11     | 68.53 +- 0.17     | 68.40 +- 0.14      | 40.42 +- 1.15          | 40.66 +- 0.97           |       
| Ours (N=100) |          17.4139  | 89.18 +- 0.29    | 91.76 +- 1.28     | 67.17 +- 2.79     | 67.27 +- 2.68      | 40.84 +- 5.36          | 41.33 +- 5.74

`./reproduce 500` will produce the following table (which is the same as what we report in the paper):

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   |
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=10) |            1.73435 | 88.47 +- 1.33    | 91.53 +- 1.62     | 66.49 +- 3.08     | 66.63 +- 3.14      | 38.33 +- 4.98          | 38.33 +- 5.22           |
| Ours (N=100) |          17.4139  | 88.45 +- 1.46    | 91.61 +- 1.71     | 66.42 +- 3.27     | 66.56 +- 3.28      | 36.56 +- 6.70          | 36.37 +- 6.97


To try different sample sizes, simply change the `--n_samples` argument in the call to `search.py`, and update the list of sample sizes on line 51 of `process_results.py`.

Note that search times may vary from the reported result owing to hardware setup.

The code is licensed under the MIT licence.

## Acknowledgements

This repository makes liberal use of code from the [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) library. We also rely on [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).
