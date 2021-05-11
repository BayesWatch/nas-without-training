# Neural Architecture Search Without Training 

:warning: Note: this repository has been updated to reflect the [second version](https://arxiv.org/abs/2006.04647) of the paper 
For the [original version of the paper](https://arxiv.org/abs/2006.04647v1), refer to the tag [v1.0](https://github.com/BayesWatch/nas-without-training/releases/tag/v1.0).:warning:

## Usage 

Create a conda environment using the env.yml file

```bash
conda env create -f env.yml
```

Activate the environment and follow the instructions to install

Install nasbench (see https://github.com/google-research/nasbench)

Download the NDS data from https://github.com/facebookresearch/nds and place the json files in naswot-codebase/nds_data/
Download the NASbench101 data (see https://github.com/google-research/nasbench)
Download the NASbench201 data (see https://github.com/D-X-Y/NAS-Bench-201)

Reproduce all of the results by running 

```bash
./scorehook.sh
```

The code is licensed under the MIT licence.

## Citing us

If you use or build on our work, please consider citing us:

```bibtex
@inproceedings{mellor2021neural,
    title={Neural Architecture Search without Training},
    author={Joseph Mellor and Jack Turner and Amos Storkey and Elliot J. Crowley},
    year={2021},
    booktitle={International Conference on Machine Learning}
}
```
