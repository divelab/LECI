# Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization

[![arXiv](https://img.shields.io/badge/arXiv-2306.01103-b31b1b.svg)](https://arxiv.org/abs/2306.01103)
[![License][license-image]][license-url]

> This is a developing repo for the implementation of our work "Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization".

[license-url]: https://github.com/divelab/LECI/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg


## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Run LECI](#run-leci)
* [Citing LECI](#citing-leci)
* [License](#license)
* [Contact](#contact)

## Overview

In this work, we propose to simultaneously incorporate label and environment causal independence (LECI) to 
**fully make use of label and environment information**, thereby addressing the challenges faced by prior methods on identifying 
causal/invariant subgraphs. We further develop an adversarial training strategy to jointly optimize these two properties for 
causal subgraph discovery with theoretical guarantees.


## Installation 

### Conda dependencies

LECI depends on [PyTorch (>=1.6.0)](https://pytorch.org/get-started/previous-versions/), [PyG (>=2.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and
[RDKit (>=2020.09.5)](https://www.rdkit.org/docs/Install.html). For more details: [conda environment](/../../blob/main/environment.yml)

> Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.

### Project installation

```shell
git clone https://github.com/divelab/LECI.git && cd LECI
pip install -e .
```

## Run LECI

```shell
goodtg --config_path final_configs/GOODHIV/scaffold/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODHIV/size/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/LBAPcore/assay/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODMotif/basis/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODMotif/size/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODCMNIST/color/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODSST2/length/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
goodtg --config_path final_configs/GOODTwitter/length/covaraite/LECI.yaml --exp_round [1/2/3/4/5/6/7/8/9/10] --gpu_idx [0..9]
```

The code is slightly optimized, so the performance may be higher than the results reported, especially for GOODHIV.
The changes will be reflected in the next version.

Explanations of the arguments can be found in this [file](/../../blob/main/GOOD_configs/GOODMotif/basis/covaraite/LECI.yaml).

### How to train LECI?

**Valid LECI:** The training of LECI is valid only when the optimal discriminator [Proposition 3.2](https://arxiv.org/pdf/2306.01103.pdf) is approximately learned, *e.g.*, 
the environment branch loss at least should not indicate a random prediction when the adversarial training is not applied (or is weak). Note that the adversarial intensity
increases from 0 to $\lambda_{EA}$ as the training proceeds, which is controlled by `self.config.train.alpha` in the code. 


**How to select the right learning rate?** Since the environment labels $E$ are noisier than normal classification labels, LECI starts with lower learning rates than general GNNs.

**How to select the valid hyperparameters?** If the EA/LA loss never decreases (invalid LECI), please try decreasing $\lambda_{EA}$ and $\lambda_{LA}$.

## Citing LECI
If you find this repository helpful, please cite our [preprint](https://arxiv.org/abs/2306.01103).
```
@article{gui2023joint,
  title={Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization},
  author={Gui, Shurui and Liu, Meng and Li, Xiner and Luo, Youzhi and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2306.01103},
  year={2023}
}
```

## License

- The GOOD datasets are under [MIT license](https://drive.google.com/file/d/1xA-5q3YHXLGLz7xV2tT69a9dcVmiJmiV/view?usp=sharing).
- The DrugOOD dataset is under [GPLv3](https://github.com/tencent-ailab/DrugOOD/blob/main/LICENSE)
- The LECI code are under [GPLv3 license](/../../blob/main/LICENSE), since the code architecture is based on [GOOD](https://github.com/divelab/GOOD.git).

## Discussion

Please submit [new issues](/../../issues/new) or start [a new discussion](/../../discussions/new) for any technical or other questions.

## Contact

Please feel free to contact [Shurui Gui](mailto:shurui.gui@tamu.edu) or [Shuiwang Ji](mailto:sji@tamu.edu)!

