# Neural Architecture Search Without Training

**IMPORTANT** : our codebase relies on use of the NASBench-201 dataset. As such we make use of cloned code from [this repository](https://github.com/D-X-Y/AutoDL-Projects). We have left the copyright notices in the code that has been cloned, which includes the name of the author of the open source library that our code relies on.

The datasets can also be downloaded as instructed from the NASBench-201 README: [https://github.com/D-X-Y/NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).

To reproduce our results:

```
conda env create -f environment.yml

conda activate nas-wot
./reproduce.sh
```

For a quick run you can set `--n_runs 3` to get results after 3 runs:

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   |
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=10)  |           1.73435 | 88.99 $\pm$ 0.24 | 92.42 $\pm$ 0.33  | 67.86 $\pm$ 0.49  | 67.54 $\pm$ 0.75   | 41.16 $\pm$ 2.31       | 40.98 $\pm$ 2.72        |
| Ours (N=100) |          17.4139  | 89.18 $\pm$ 0.29 | 91.76 $\pm$ 1.28  | 67.17 $\pm$ 2.79  | 67.27 $\pm$ 2.68   | 40.84 $\pm$ 5.36       | 41.33 $\pm$ 5.74

The size of `N` is set with `--n_samples 10`. To produce the results in the paper, set `--n_runs 500`:

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   |
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=10)  |           1.73435 | 89.25 $\pm$ 0.08 | 92.21 $\pm$ 0.11  | 68.53 $\pm$ 0.17  | 68.40 $\pm$ 0.14   | 40.42 $\pm$ 1.15       | 40.66 $\pm$ 0.97        |
| Ours (N=100) |          17.4139  | 88.45 $\pm$ 1.46 | 91.61 $\pm$ 1.71  | 66.42 $\pm$ 3.27  | 66.56 $\pm$ 3.28   | 36.56 $\pm$ 6.70       | 36.37 $\pm$ 6.97


The code is licensed under the MIT licence.
