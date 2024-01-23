"""Script to run and evaluate the few-shot learning experiment with uncertainty."""
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import tensorflow_datasets as tfds
import ml_collections.config_flags

from absl import app, flags, logging
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from src.bpnn.curvature_scalings import ScalarScaling, StandardBayes
from src.bpnn.utils import base_path, device, set_seed

from tools.fewshot import ImageNetKaggle
from tools.run_experiment import sweep_fewshot_bpnns

ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, 
                    help='Work unit directory.')
flags.DEFINE_string('imagenet_root', default=None, 
                    help='Root to imagenet prior.')
flags.DEFINE_string('datasetname', default="cifar100", 
                    help='Name of the dataset.')
flags.DEFINE_string('dataload_dir', default=None, 
                    help='Work unit directory.')
flags.DEFINE_string('tpu', None, 
                    'Unused. Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_integer('num_cores', default=None,
                     help='Unused. How many devices being used.')
flags.DEFINE_integer('shot', default=5, 
                    help='How many shots?')
flags.DEFINE_integer('num_epochs', default=100, 
                    help='How many epochs?')
flags.DEFINE_integer('num_batch', default=8, 
                    help='Batch size')
flags.DEFINE_boolean('use_gpu', default=None, 
                    help='Unused. Whether or not running on GPU.')
flags.DEFINE_boolean('feature_extract', default=True, 
                    help='Whether or not fine tuning only last or full.')
flags.DEFINE_integer('id', default=0, 
                    help='the id of the experiments')
flags.DEFINE_string('type', default="PBNN", 
                    help='one of PBNN, PNN, and evaluate')
flags.DEFINE_string('curvature_type', default="KFAC", 
                    help='one of KFOC, KFAC, and Diagonal (only considered if type=PBNN)')
flags.DEFINE_string('criterion_type', default="McAllester", 
                    help='one of MAP, McAllester, and Catoni (only considered if type=PBNN)')
flags.DEFINE_string('curvature_scaling', default="McAllester", 
                    help='one of MAP, McAllester, and Catoni (only considered if type=PBNN)')
flags.DEFINE_boolean('isotropic_prior', default=False, 
                    help='whether the isotropic or learned prior is used (only considered if type=PBNN)')
flags.DEFINE_float('temperature_scaling', default=1e-8, 
                    help='the temperature scaling (only considered if type=PBNN and curvature_scaling=Validation)')
FLAGS = flags.FLAGS


def main(FLAGS):
    set_seed(9768940)

    # initializations
    def write_note(note):
        logging.info('NOTE: %s', note)
    write_note('Initializing...')
    last_layer_name = 'fc'
    lateral_connections = ['layer1.0.downsample.0',
                           'layer2.0.downsample.0',
                           'layer3.0.downsample.0',
                           'layer4.0.downsample.0']
    learning_rate = 1e-4
    confidence = 0.8
    len_data = 1281167
    negative_data_log_likelihood = 575557.7385101318
    torch.hub.set_dir(os.path.join(base_path, os.pardir, 'torch_hub'))
    network = resnet50(weights='IMAGENET1K_V1')

    # checkout the data loader for few-shot learning
    write_note('Loading the datasets...')
    
    # pbnn data loader
    transforms = Compose([ToTensor(), Resize(256), CenterCrop(224),
                          Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])
    train_ds = ImageNetKaggle(root=FLAGS.imagenet_root,
                              split='train',
                              transform=transforms)
    test_ds = ImageNetKaggle(root=FLAGS.imagenet_root,
                             split='val',
                             transform=transforms)
    train_ds = torch.utils.data.DataLoader(
        train_ds, batch_size=FLAGS.num_batch, shuffle=True,
        num_workers=8, pin_memory=False
    )
    test_ds = torch.utils.data.DataLoader(
        test_ds, batch_size=FLAGS.num_batch, shuffle=False,
        num_workers=8, pin_memory=False
    )
    prior_set = (1000, tuple([train_ds, test_ds, test_ds]), torch.nn.CrossEntropyLoss())

    # construct dataset loaders in tuple format
    dataloaders = [prior_set]
    write_note('Sweeping the PBNN...')
    if FLAGS.type == 'PBNN':
        if FLAGS.curvature_scaling == 'MAP':
            FLAGS.curvature_scaling = 'Validation'
        name = 'isotropic' if FLAGS.isotropic_prior else 'learned'
        prior_curvature_scalings = [ScalarScaling(1., 0., 1.)] if FLAGS.isotropic_prior else \
            [StandardBayes()] if FLAGS.curvature_scaling == 'Validation' else \
                None
        if name == 'isotropic':
            weight_decays=[1e-10, 1e-8, 1e-6]
            FLAGS.temperature_scaling=[1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24]
        else:
            weight_decays=[1e-10, 1e-8, 1e-6]
            FLAGS.temperature_scaling=[1e-14, 1e-16, 1e-18, 1e-20, 1e-22, 1e-24]

        # sweep pbnn initiate
        sweep_fewshot_pbnns(prefix=os.path.join('fewshot_uncertainty', FLAGS.datasetname, f'{FLAGS.id}_{name}_'),
                            dataloaders=dataloaders,
                            network=network,
                            backbone=None,
                            last_layer_name=last_layer_name,
                            lateral_connections=lateral_connections,
                            curvature_types=[FLAGS.curvature_type],
                            weight_decays=weight_decays,
                            temperature_scalings=FLAGS.temperature_scaling,
                            evaluate_each_temperature=True,
                            criterion_types=[FLAGS.criterion_type],
                            curvature_scalings=[FLAGS.curvature_scaling],
                            pretrained=True,
                            curvature_device=device,
                            prior_curvature_scalings=prior_curvature_scalings,
                            learning_rate=learning_rate,
                            prior_curvature_path=os.path.join(base_path, 'models', f'resnet50_imagenet_{FLAGS.curvature_type}.pt'),
                            negative_data_log_likelihood=negative_data_log_likelihood,
                            len_data=len_data,
                            confidence=confidence,
                            curvature_scaling_device=device if FLAGS.curvature_scaling == 'Validation' else torch.device('cpu'),
                            compute_pac_bounds=False,
                            alphas_betas=([1.], [1.]),
                            FLAGS=FLAGS,
                            validation_scaling_num_samples=10
                            )
    else:
        raise AttributeError


if __name__ == '__main__':
    def _main(argv):
        del argv
        main(FLAGS)

    app.run(_main)  # Ignore the returned values from `main`.
