"""Script to run and evaluate the bayesian continual learning experiment for Table 1."""
import argparse
from os import pardir
from os.path import join

import torch
from torchvision.models import resnet50

from src.bpnn.curvature_scalings import ScalarScaling, StandardBayes
from src.bpnn.utils import base_path, device
from tools.evaluate_experiment import extract_df_continual_learning
from tools.run_experiment import sweep_bpnn, sweep_dpnn, sweep_pnn
from tools.wrgbd_dataloaders import get_dataloaders

"""
All experiments of the bayesian continual learning experiment can be run by executing the following commands:

This command runs all Progressive Neural Networks with and without MC droput:
python bayesian_continual_learning.py --id 0 --type PNN

This command runs Bayesian Progressive Neural Networks with a zero mean isotropic prior:
python bayesian_continual_learning.py --id 0 --type BPNN --curvature_type KFOC --criterion_type MAP --curvature_scaling Validation --isotropic_prior --mean0 --temperature_scaling 1e-12 1e-16 1e-20 1e-24 1e-28

This command runs Bayesian Progressive Neural Networks with an isotropic prior with a learned mean:
python bayesian_continual_learning.py --id 0 --type BPNN --curvature_type KFOC --criterion_type MAP --curvature_scaling Validation --isotropic_prior --temperature_scaling 1e-12 1e-16 1e-20 1e-24 1e-28

This command runs Bayesian Progressive Neural Networks with a learned prior:
python bayesian_continual_learning.py --id 0 --type BPNN --curvature_type KFOC --criterion_type MAP --curvature_scaling Validation --temperature_scaling 1e-12 1e-16 1e-20 1e-24 1e-28

After running all experiments (for multiple ids if multiple random states are used), the results can be evaluated by executing the following command:
python bayesian_continual_learning.py --type evaluate
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Progressive Neural Networks')
    parser.add_argument('--id', nargs='?', default=0, help='the id of the experiment')
    parser.add_argument('--type', help='one of BPNN, PNN, and evaluate')
    parser.add_argument(
        '--curvature_type', nargs='?',
        help='one of KFOC, KFAC, and Diagonal (only considered if type=BPNN)')
    parser.add_argument(
        '--criterion_type', nargs='?',
        help='one of MAP, McAllester, and Catoni (only considered if type=BPNN)')
    parser.add_argument(
        '--curvature_scaling', nargs='?',
        help='one of Standard, McAllester, and Catoni (only considered if type=BPNN)')
    parser.add_argument(
        '--isotropic_prior', default=False, action="store_true",
        help='whether the isotropic or learned prior is used (only considered if type=BPNN)')
    parser.add_argument(
        '--mean0', default=False, action="store_true",
        help='whether the isotropic prior should have mean 0 (only considered if type=BPNN)')
    parser.add_argument(
        '--temperature_scaling', nargs='*', type=float,
        help='the temperature scaling (only considered if type=BPNN and curvature_scaling=Validation)')
    args = parser.parse_args()

    torch.hub.set_dir(join(base_path, pardir, 'torch_hub'))
    network = resnet50(weights='IMAGENET1K_V1')

    last_layer_name = 'fc'
    lateral_connections = ['layer1.0.downsample.0',
                           'layer2.0.downsample.0',
                           'layer3.0.downsample.0',
                           'layer4.0.downsample.0']

    learning_rate = 1e-4
    confidence = 0.8

    len_data = 1281167
    negative_data_log_likelihood = 575557.7385101318

    dataloaders = get_dataloaders(batch_size=4, remove_idxs=[0])

    if args.type == 'BPNN':
        # temperature_scaling = 1e-8 if args.curvature_type in ['KFAC', 'KFOC'] else 1e-10
        name = 'isotropic' if args.isotropic_prior and not args.mean0 else 'mean0' if args.mean0 else 'learned'
        prior_curvature_scalings = [ScalarScaling(1., 0., 1.)] if args.isotropic_prior else \
            [StandardBayes()] if args.curvature_scaling == 'Validation' else \
                None

        sweep_bpnn(
            prefix=join('continual_learning', 'large_scale3', f'{args.id}_{name}_'),
            dataloaders=dataloaders,
            network=network,
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            curvature_types=[args.curvature_type],
            weight_decays=[1e-8],
            temperature_scalings=args.temperature_scaling,
            evaluate_each_temperature=False,
            eval_every_task=False,
            criterion_types=[args.criterion_type],
            curvature_scalings=[args.curvature_scaling],
            pretrained=True,
            curvature_device=device,
            isotropic_prior=args.mean0,
            prior_curvature_scalings=prior_curvature_scalings,
            learning_rate=learning_rate,
            prior_curvature_path=join(base_path, 'models', f'resnet50_imagenet_{args.curvature_type}.pt'),
            negative_data_log_likelihood=negative_data_log_likelihood,
            len_data=len_data,
            confidence=confidence,
            curvature_scaling_device=device if args.curvature_scaling == 'Validation' else torch.device('cpu'),
            compute_pac_bounds=False,
            alphas_betas=([1.], [1.]),
            validation_scaling_num_samples=10,
            )
    elif args.type == 'PNN':
        for id in range(3):
            sweep_pnn(
                prefix=join('continual_learning', 'large_scale3', f'{args.id}_'),
                dataloaders=dataloaders,
                network=network,
                backbone=None,
                last_layer_name=last_layer_name,
                lateral_connections=lateral_connections,
                weight_decays=[1e-3, 1e-5, 1e-8],
                pretrained=True,
                learning_rate=learning_rate,
                eval_every_task=False,
                )

            dropout_positions = ['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.0.conv3', 'layer1.0.downsample.0',
                                 'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3', 'layer1.2.conv1',
                                 'layer1.2.conv2',
                                 'layer1.2.conv3', 'layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.conv3',
                                 'layer2.0.downsample.0', 'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3',
                                 'layer2.2.conv1', 'layer2.2.conv2', 'layer2.2.conv3', 'layer2.3.conv1',
                                 'layer2.3.conv2',
                                 'layer2.3.conv3', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3',
                                 'layer3.0.downsample.0', 'layer3.1.conv1', 'layer3.1.conv2', 'layer3.1.conv3',
                                 'layer3.2.conv1', 'layer3.2.conv2', 'layer3.2.conv3', 'layer3.3.conv1',
                                 'layer3.3.conv2',
                                 'layer3.3.conv3', 'layer3.4.conv1', 'layer3.4.conv2', 'layer3.4.conv3',
                                 'layer3.5.conv1',
                                 'layer3.5.conv2', 'layer3.5.conv3', 'layer4.0.conv1', 'layer4.0.conv2',
                                 'layer4.0.conv3',
                                 'layer4.0.downsample.0', 'layer4.1.conv1', 'layer4.1.conv2', 'layer4.1.conv3',
                                 'layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3']

            sweep_dpnn(
                prefix=join('continual_learning', 'large_scale3', f'{args.id}_'),
                dataloaders=dataloaders,
                network=network,
                backbone=None,
                last_layer_name=last_layer_name,
                lateral_connections=lateral_connections,
                weight_decays=[0., 1e-5],
                dropout_probabilities=[.01, .1, .3],
                dropout_positions=dropout_positions,
                pretrained=True,
                learning_rate=learning_rate,
                eval_every_task=False,
                )

    # # EVALUATE
    elif args.type == 'evaluate':
        root_path = join(base_path, 'results', 'continual_learning', 'large_scale3')
        target_path = join(base_path, 'reports')

        df_mean, df_std = extract_df_continual_learning(root_path)

        df_mean.to_csv(join(target_path, 'bayesian_continual_learning_mean.csv'))
        df_std.to_csv(join(target_path, 'bayesian_continual_learning_std.csv'))
