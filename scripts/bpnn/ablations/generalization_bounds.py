"""Script to run and evaluate the generalization bounds for the table in Figure 2 (a)."""
from os import makedirs
from os.path import join

from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.curvature_scalings import StandardBayes
from src.bpnn.utils import torch_data_path, base_path
from src.curvature.lenet5 import lenet5
from tools.evaluate_experiment import full_dict, extract_df_transfer_learning
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
        num_workers=1
    ), CrossEntropyLoss()) for dataset_class in [MNIST, NotMNIST]]

    learning_rate = 5e-4
    confidence = 0.9

    # training
    for i in range(5):
        # learned prior, curvature scaling, frequentist projection
        sweep_bpnn(
            prefix=join('ablations', 'generalization_bounds', f'{i}_empirical_prior_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-8, 1e-5, 1e-3],
            temperature_scalings=[10 ** (-i) for i in range(26)],
            criterion_types=['MAP', 'MAP', 'MAP', 'McAllester', 'Catoni'],
            curvature_scalings=['Standard', 'McAllester', 'Catoni', 'McAllester', 'Catoni'],
            pretrained=True,
            curvature_device=None,
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
            curvature_device=None,
            isotropic_prior=True,
            learning_rate=learning_rate,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

        # grid search
        sweep_bpnn(
            prefix=join('ablations', 'generalization_bounds', f'{i}_empirical_prior_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-8, 1e-5, 1e-3],
            temperature_scalings=[10 ** (-i) for i in range(26)],
            criterion_types=['MAP'],
            curvature_scalings=['Validation'],
            pretrained=True,
            curvature_device=None,
            learning_rate=learning_rate,
            confidence=confidence,
            prior_curvature_scalings=[StandardBayes()],
            evaluate_each_temperature=True,
            validation_scaling_num_samples=10,
            alphas_betas=([0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 1., 2.], [0.01, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 1., 2.])
            )

        # evaluation and visualization
        root_path = join(base_path, 'results', 'ablations', 'generalization_bounds')
        target_path = join(base_path, 'reports', 'ablations', 'transfer_learning')
        makedirs(target_path, exist_ok=True)

        d = full_dict(root_path)
        df = extract_df_transfer_learning(d)
