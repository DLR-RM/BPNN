"""Script to run and evaluate the small-scale cold posterior experiment for Figure 2 (c) / Figure 8 left."""

from os import makedirs
from os.path import join

from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.curvature_scalings import ScalarScaling
from src.bpnn.utils import device
from src.bpnn.utils import torch_data_path, base_path
from src.curvature.lenet5 import lenet5
from tools.evaluate_experiment import extract_df_transfer_learning
from tools.evaluate_experiment import full_dict, cold_posterior_plot
from tools.mnist_dataloaders import get_train_val_test_split_dataloaders
from tools.not_mnist import NotMNIST
from tools.run_experiment import sweep_bpnn

if __name__ == '__main__':
    # configuration
    network = lenet5(pretrained=True)

    last_layer_name = '11'
    lateral_connections = ['3', '7', '9']

    dataloaders = [(10, get_train_val_test_split_dataloaders(
        dataset_class=dataset_class,
        torch_data_path=torch_data_path,
        split=[True, False],
        transform=ToTensor(),
        batch_size=256,
        num_workers=1,
    ), CrossEntropyLoss()) for dataset_class in [MNIST, NotMNIST]]

    learning_rate = 5e-4
    confidence = 0.2

    # training with 25 different initializations
    for i in range(25):
        # learned prior
        sweep_bpnn(
            prefix=join('ablations', 'cold_posterior_small', f'{i}_learned_prior_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(10)] + [0.3],
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
            prefix=join('ablations', 'cold_posterior_small', f'{i}_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(10)] + [0.3],
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
            prefix=join('ablations', 'cold_posterior_small', f'{i}_zero_mean_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(10)] + [0.3],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=device,
            isotropic_prior=True,
            learning_rate=learning_rate,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

    # evaluation and visualization
    root_path = join(base_path, 'results', 'ablations', 'cold_posterior_small')
    target_path = join(base_path, 'reports', 'ablations', 'transfer_learning')
    makedirs(target_path, exist_ok=True)

    d = full_dict(root_path)
    df = extract_df_transfer_learning(d)

    df_curv = df[~df['empirically learned prior'].isna() &
                 (df['curvature scaling'] == 'Standard') &
                 (df['curvature type'] == 'KFOC')]
    f = cold_posterior_plot(df_curv)
    f.savefig(join(target_path, 'cold_posterior_small.png'))
    f.show()
