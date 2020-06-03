# Neural Architecture Search Without Training

**IMPORTANT** : our codebase relies on use of the NASBench-201 dataset. As such we make use of cloned code from [this repository](https://github.com/D-X-Y/AutoDL-Projects). We have left the copyright notices in the code that has been cloned, which includes the name of the author of the open source library that our code relies on.

The datasets can also be downloaded as instructed from the NASBench-201 README: [https://github.com/D-X-Y/NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201).

To exactly reproduce our results:

```
conda env create -f environment.yml

conda activate nas-wot
./reproduce.sh
```
