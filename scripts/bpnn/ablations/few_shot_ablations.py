"""Script to run and evaluate the ablations on the few-shot accuracy experiment for Figure 2 (b) / Figure 7."""
from os import makedirs
from os.path import join

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.curvature_scalings import ScalarScaling
from src.bpnn.utils import base_path
from src.bpnn.utils import torch_data_path
from src.curvature.lenet5 import lenet5
from tools.evaluate_experiment import extract_df_transfer_learning
from tools.evaluate_experiment import full_dict
from tools.mnist_dataloaders import get_train_val_test_split_dataloaders
from tools.not_mnist import NotMNIST
from tools.run_experiment import sweep_bpnn

if __name__ == '__main__':
    # configuration
    network = lenet5(pretrained=True)

    last_layer_name = '11'
    lateral_connections = ['3', '7', '9']

    mnist_dataloader, not_mnist_dataloader = [(10, get_train_val_test_split_dataloaders(
        dataset_class=dataset_class,
        torch_data_path=torch_data_path,
        split=[True, False],
        transform=ToTensor(),
        batch_size=256,
        num_workers=1
    ), CrossEntropyLoss()) for dataset_class in [MNIST, NotMNIST]]

    learning_rate = 5e-4
    confidence = 0.9

    num_classes, (dataloader_train, dataloader_val, dataloader_test), loss = not_mnist_dataloader
    targets_train = np.array([target for _, target in dataloader_train.dataset])
    indices_per_class = [np.where(targets_train == i)[0] for i in range(num_classes)]

    # training
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, None]:
        indices = np.concatenate([np.random.permutation(indices)[:i] for indices in indices_per_class])
        small_train = torch.utils.data.Subset(dataloader_train.dataset, indices)
        small_dataloader_train = torch.utils.data.DataLoader(
            small_train,
            pin_memory=dataloader_train.pin_memory,
            num_workers=dataloader_train.num_workers,
            batch_size=dataloader_train.batch_size,
            shuffle=True)
        dataloaders = [mnist_dataloader,
                       (num_classes,
                        (small_dataloader_train, dataloader_val, dataloader_test),
                        loss)]

        # learned prior
        sweep_bpnn(
            prefix=join('ablations', 'few_shot_ablations', f'{i}_learned_prior_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[10 ** (-i) for i in range(10)],
            temperature_scalings=[10 ** (-i) for i in range(20)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=None,
            learning_rate=learning_rate,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

        # isotropic prior with learned mean
        sweep_bpnn(
            prefix=join('ablations', 'few_shot_ablations', f'{i}_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[10 ** (-i) for i in range(10)],
            temperature_scalings=[10 ** (-i) for i in range(20)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=None,
            prior_curvature_scalings=[ScalarScaling(1., 0., 1.)],
            learning_rate=learning_rate,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

        # isotropic prior with zero mean
        sweep_bpnn(
            prefix=join('ablations', 'few_shot_ablations', f'{i}_zero_mean_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[10 ** (-i) for i in range(10)],
            temperature_scalings=[10 ** (-i) for i in range(20)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=None,
            isotropic_prior=True,
            learning_rate=learning_rate,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

    # evaluation and visualization
    root_path = join(base_path, 'results', 'ablations', 'few_shot_ablations')
    target_path = join(base_path, 'reports', 'ablations', 'transfer_learning')
    makedirs(target_path, exist_ok=True)

    d = full_dict(root_path)
    df = extract_df_transfer_learning(d)

    df['num samples'] = df.apply(
        lambda row:
        int(row.name.split('_')[0]) if row.name.split('_')[0] != 'None' else
        5400, axis=1)

    f, axs = plt.subplots(10, 20, sharex=True, sharey=True, figsize=(30, 15))
    for k, j in enumerate(range(10)):
        for i in range(20):
            ax = axs[k][i]
            standard_bayes = df[
                (df['curvature type'] == 'KFOC') & (df['criterion'] == 'MAP') & (
                        df['curvature scaling'] == 'Standard') & (
                        df['weight decay'] == 10 ** (-j)) & (df['temperature scaling'] == 10 ** (-i))]
            sns.lineplot(
                data=standard_bayes, x='num samples', y='accuracy', hue='prior type',
                hue_order=['learned', 'isotropic', 'isotropic mean zero'],
                ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    for ax in axs.flat:
        ax.legend([], [], frameon=False)
        ax.set_xscale('log')
    for i, ax in enumerate(axs[0]):
        ax.set_title(f'{10 ** (-i)}')
    for j, ax in enumerate(axs[:, 0]):
        ax.annotate(
            f'{10 ** (-j)}',
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 5, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
        )
    f.legend(handles, labels)
    f.tight_layout()
    f.savefig(join(target_path, 'few_shot_ablations.png'))
    f.show()
