"""Script to run and evaluate the large-scale cold posterior experiment for Figure 8 right."""
from os import makedirs
from os import pardir
from os.path import join

import torch
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

from src.bpnn.curvature_scalings import ScalarScaling
from src.bpnn.utils import torch_data_path, base_path, device
from tools.evaluate_experiment import full_dict, extract_df_transfer_learning, cold_posterior_plot
from tools.mnist_dataloaders import get_train_val_test_split_dataloaders
from tools.run_experiment import sweep_bpnn

if __name__ == '__main__':
    torch.hub.set_dir(join(base_path, pardir, 'torch_hub'))

    # configuration
    network = resnet50(weights='IMAGENET1K_V1')

    last_layer_name = 'fc'
    lateral_connections = [
        'layer1.0.downsample.0',
        'layer2.0.downsample.0',
        'layer3.0.downsample.0',
        'layer4.0.downsample.0'
    ]

    cifar10_dataloader = get_train_val_test_split_dataloaders(
        dataset_class=CIFAR10,
        torch_data_path=torch_data_path,
        split=[True, False],
        transform=Compose(
            [
                Resize([224, 224]),
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ]),
        batch_size=8
    )
    dataloaders = [
        (1000, ([], [], []), torch.nn.CrossEntropyLoss()),
        (10, cifar10_dataloader, torch.nn.CrossEntropyLoss())
    ]

    len_data = 1281167
    negative_data_log_likelihood = 575557.7385101318

    learning_rate = 1e-3
    confidence = 0.8
    num_epochs = 100

    # training with 5 different initializations
    for i in range(5):
        # learned prior
        sweep_bpnn(
            prefix=join('ablations', 'cold_posterior_large', f'{i}_learned_prior_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(8, 16)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            prior_curvature_path=join(base_path, 'models', f'resnet50_imagenet_KFOC.pt'),
            negative_data_log_likelihood=negative_data_log_likelihood,
            len_data=len_data,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

        # isotropic prior with learned mean
        sweep_bpnn(
            prefix=join('ablations', 'cold_posterior_large', f'{i}_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(8, 16)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=device,
            prior_curvature_scalings=[ScalarScaling(1., 0., 1.)],
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            prior_curvature_path=join(base_path, 'models', f'resnet50_imagenet_KFOC.pt'),
            negative_data_log_likelihood=negative_data_log_likelihood,
            len_data=len_data,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

        # isotropic prior with zero mean
        sweep_bpnn(
            prefix=join('ablations', 'cold_posterior_large', f'{i}_zero_mean_isotropic_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=['KFOC'],
            weight_decays=[1e-3, 1e-5, 1e-8],
            temperature_scalings=[10 ** (-i) for i in range(8, 16)],
            criterion_types=['MAP'],
            curvature_scalings=['Standard'],
            pretrained=True,
            curvature_device=device,
            isotropic_prior=True,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            prior_curvature_path=join(base_path, 'models', f'resnet50_imagenet_KFOC.pt'),
            negative_data_log_likelihood=negative_data_log_likelihood,
            len_data=len_data,
            confidence=confidence,
            evaluate_each_temperature=True,
        )

    # evaluation and visualization
    root_path = join(base_path, 'results', 'ablations', 'cold_posterior_large')
    target_path = join(base_path, 'reports', 'ablations', 'transfer_learning')
    makedirs(target_path, exist_ok=True)

    d = full_dict(root_path)
    df = extract_df_transfer_learning(d)

    df_curv = df[~df['empirically learned prior'].isna() &
                 (df['curvature scaling'] == 'Standard') &
                 (df['curvature type'] == 'KFOC')]
    f = cold_posterior_plot(df_curv)
    f.savefig(join(target_path, 'cold_posterior_large.png'))
    f.show()
