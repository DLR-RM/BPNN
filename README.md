# Bayesian Progressive Neural Networks

Bayesian Progressive Neural Networks (BPNN) is the official code base for the following works:

**[Kronecker-Factored Optimal Curvature]** \
Dominik Schnaus, Jongseok Lee, and Rudolph Triebel. _NeurIPS 2021 BDL Workshop_.

**[Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks]** \
Dominik Schnaus, Jongseok Lee, Daniel Cremers, Rudolph Triebel. ICML 2023.


## Description
### Kronecker-Factored Optimal Curvature

The Kronecker-Factored Optimal Curvature (K-FOC) improves the approximation quality of the Fisher Information Matrix (FIM)
as a Kronecker-factored matrix compared to the widely-adopted Kronecker-Factored Approximate Curvature (K-FAC)[^1]. 
For this, we adapted the power method to find the optimal Kronecker factors for a batch of samples where each step has
a similar complexity as K-FAC.


### Bayesian Progressive Neural Networks

Bayesian Progressive Neural Networks (BPNN) are a probabilistic architecture that combines Laplace approximation[^2]
with Progressive Neural Networks (PNN)[^3]. In a first step, a prior distribution is learned from a different dataset and
is used as a prior for the main columns. The lateral connections each use the posterior of the outgoing layer as a 
prior. This architecture leads to a model that can use previously learned features and priors to improve the performance
on the later tasks.

Altogether, this implementation also includes a general version of PNN that can use arbitrary 
network architectures and an implementation of PNN that includes MC Dropout[^4].


## Installation

You can install this project either with conda or the Python Package Installer (pip).

1. Clone or download the repository
2. Install the packages either with conda or with pip:
   - conda:
     - Create a new environment called `bpnn`
     ```bash
     conda create -n bpnn python=3.8
     conda activate bpnn
     ```
     - Install [PyTorch and Torchvision](https://pytorch.org/get-started/locally/)
     - Go into the root folder and install the remaining packages:
     ```bash
     conda env update --file environment.yml
     ```
   - pip:
     - Install [PyTorch and Torchvision](https://pytorch.org/get-started/locally/)
     - Go into the root folder and install the remaining packages:
     ```bash
     pip install -e .
     ```

## Reproducing the Results
Will be added soon.

## Usage

### Kronecker-Factored Optimal Curvature

To use the curvatures, please follow the steps of the [official curvature library] where this curvature 
implementation is forked from. K-FOC can be used the same as K-FAC. Additionally, `src/bpnn/utils.py` 
includes `compute_curvature` which contains the loop to compute the curvature for a model over a dataset.

### Bayesian Progressive Neural Networks

To run own BPNN experiments, one can use the functions `sweep_bpnn`, `sweep_pnn`, and `sweep_dpnn` in
`tools/run_experiment.py` that train multiple configurations of Bayesian Progressive Neural Networks, 
Progressive Neural Networks, and Progressive Neural Networks with MC Dropout.

These functions take the dataloaders, the base network, the names of the layers for lateral connections,
and the name of the last layer. Moreover, multiple different weight decays, curvature types, etc. can be 
specified and all combinations of these values are trained and the results are saved for each model 
individually in the `results` folder.

The functions in `tools/evaluate_experiment.py` can be used to evaluate and plot these results.

Please see the docstrings of the functions for more information.
Examples of a full training and evaluation can be found in `scripts/bpnn/`.

## Project Organization

```
├── LICENSE.txt                             <- The GNU General Public License.
├── README.md                               <- The top-level README.
├── data                                    <- The datasets used in the experiments.
│   ├── raw                                 <- Raw data files for
│   │   ├── Concrete_Data.xls               <-     the Concrete Compression Strength Dataset and
│   │   └── ENB2012_data.xlsx               <-     the Energy Efficiency Dataset.
│   └── torch                               <- All other datasets are automatically downloaded and 
│                                                saved here.
├── environment.lock.yml                    <- The exact conda environment file for reproducibility.
├── environment.yml                         <- The conda environment file with the requirements.
├── models                                  <- The pre-trained models and the models saved during the 
│   │                                           training.
│   ├── kmnist_lenet.pt                     <- LeNet5 with 3 input channels trained on Kuzushiji-MNIST.
│   ├── resnet50_imagenet_KFAC.pt           <- The K-FAC curvature of a pre-trained resnet50 on ImageNet.
│   └── resnet50_imagenet_KFOC.pt           <- The K-FOC curvature of a pre-trained resnet50 on ImageNet.
├── pyproject.toml                          <- Build system configuration.
├── reports                                 <- Generated plots and tables.
├── scripts                                 <- The scripts to reproduce the different experiments
├── setup.cfg                               <- Declarative configuration of the project.
├── setup.py
├── src                                     <- The implementation of the main functionality.
│   ├── curvature                           <- The curvature implementations 
│   │   │                                       (fork of https://github.com/DLR-RM/curvature).
│   │   ├── curvatures.py                   <- Different curvature approximations including K-FOC.
│   │   ├── lenet5.py                       <- Loading of LeNet5.
│   │   ├── lenet5_mnist.pth                <- LeNet5 with 1 input channels trained on MNIST.
│   │   └── utils.py                        <- Utilities to compute the curvatures (e.g. power method).
│   └── bpnn                                <- The Bayesian Progressive Neural Networks implementation.
│       ├── criterions.py                   <- Different criterions to optimize the weights 
│       │                                       e.g. with PAC-Bayes bounds
│       ├── curvature_scalings.py           <- Different methods to scale the curvature scales.
│       ├── bpnn.py                         <- Main implementation of BPNN and utility functions to fit it.
│       ├── pnn.py                          <- Implementation of PNN (also with MC Dropout) and the 
│       │                                       fitting of general PNN and its adaptions.
│       └── utils.py                        <- Utility functions for BPNN (e.g. metrics, training loop)
├── tests                                   <- Unit tests which can be run with `py.test`.
├── tools                                   <- Tools to run experiments.
│   ├── evaluate_experiment.py              <- Functions to evaluate the JSON files after the training.
│   ├── mnist_dataloaders.py                <- Generates the dataloaders used to train the small-scale 
│   │                                           continual learning experiment.
│   ├── not_mnist.py                        <- The NotMNIST Dataset.
│   ├── run_experiment.py                   <- Functions to run multiple configurations of BPNN and PNN.
│   ├── wrgbd.py                            <- The Washington University's RGB-D Object (WRGBD) Dataset.
│   └── wrgbd_dataloaders.pt                <- Generates the dataloaders used to train the large-scale 
│                                               continual learning experiment.
└── .coveragerc                             <- Configuration for coverage reports of unit tests.
```

## Citation
If you find this project useful, please cite us in the following ways:
```
@inproceedings{schnaus2021kronecker,
  title = {Kronecker-Factored Optimal Curvature},
  author = {Schnaus, Dominik and Lee, Jongseok and Triebel, Rudolph},
  year = {2021},
  maintitle = {Thirty-fifth Conference on Neural Information Processing Systems},
  booktitle = {Bayesian Deep Learning Workshop},
}

@inproceedings{schnaus2023,
  title = {Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks},
  author = {Schnaus, Dominik and Lee, Jongseok and Cremers, Daniel and Triebel, Rudolph},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {30252--30284},
  year = {2023},
  editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v202/schnaus23a/schnaus23a.pdf},
  url = {https://proceedings.mlr.press/v202/schnaus23a.html},
}

```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.0.2 and the [dsproject extension] 0.6.1.

## Bibliography

[^1]: Martens, James, and Roger Grosse. "Optimizing neural networks with kronecker-factored approximate curvature." 
International conference on machine learning. PMLR, 2015.

[^2]: MacKay, David JC. "A practical Bayesian framework for backpropagation networks." 
Neural computation 4.3 (1992): 448-472.

[^3]: Rusu, Andrei A., et al. "Progressive neural networks." arXiv preprint arXiv:1606.04671 (2016).

[^4]: Gal, Yarin, and Zoubin Ghahramani. 
"Bayesian convolutional neural networks with Bernoulli approximate variational inference." 
arXiv preprint arXiv:1506.02158 (2015).

[resnet50_imagenet_KFAC.pt]: https://drive.google.com/file/d/1MwMSyq6_yLU8KDrr7Fy_Lr75o95-jyxE/view?usp=sharing
[resnet50_imagenet_KFOC.pt]: https://drive.google.com/file/d/1lczWZ8jaugLPTRLxy2h5GTJ5uSzl3RS4/view?usp=sharing
[official curvature library]: https://github.com/DLR-RM/curvature
[ImageNet ILSVRC 2012]: http://www.image-net.org/download-images
[Kronecker-Factored Optimal Curvature]: http://bayesiandeeplearning.org/2021/papers/33.pdf
[Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks]: https://proceedings.mlr.press/v202/schnaus23a/schnaus23a.pdf
[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject