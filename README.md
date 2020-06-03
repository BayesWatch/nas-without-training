# Neural Architecture Search Without Training

**IMPORTANT** : our codebase relies on use of the NASBench-201 dataset. As such we make use of cloned code from [this repository](https://github.com/D-X-Y/AutoDL-Projects). We have left the copyright notices in the code that has been cloned, which includes the name of the author of the open source library that our code relies on.

The datasets can also be downloaded as instructed from the NASBench-201 README: [https://github.com/D-X-Y/NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).

To exactly reproduce our results:

```
conda env create -f environment.yml

conda activate nas-wot
./reproduce.sh
```

Will produce the following table:

| Method       |   Search time (s) | CIFAR-10 (val)   | CIFAR-10 (test)   | CIFAR-100 (val)   | CIFAR-100 (test)   | ImageNet16-120 (val)   | ImageNet16-120 (test)   |
|:-------------|------------------:|:-----------------|:------------------|:------------------|:-------------------|:-----------------------|:------------------------|
| Ours (N=100) |             18.35 | 89.18 +- 0.29    | 91.76 +- 1.28     | 67.17 +- 2.79     | 67.27 +- 2.68      | 40.84 +- 5.36          | 41.33 +- 5.74           |

The code is licensed under the MIT licence.
