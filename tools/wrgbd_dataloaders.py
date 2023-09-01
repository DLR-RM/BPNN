"""Dataloader of the large-scale continual learning experiment"""
import os
from typing import List, Tuple, Optional, Dict, Iterable

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize

from src.bpnn.utils import seed, torch_data_path
from tools.wrgbd import WRGBD


def get_train_val_test_split_dataloaders(
        dataset_class: type,
        data_path: str,
        splits: List,
        download: bool,
        transform,
        batch_size: int = 1,
        kwargs: Optional[Dict] = None) \
        -> Tuple[int, Tuple[DataLoader, ...], torch.nn.Module]:
    """Returns the training, validation and test split for a dataset.

    Args:
        dataset_class: A dataset class
        data_path: The path to the datasets
        splits: A list of [training, validation, test] for the split
            parameter of the dataset class
        download: Whether the datasets should be downloaded
        transform: A torchvision transform
        batch_size: The batch size for the dataloader
        kwargs: Keyword arguments for the dataset class

    Returns:
        The number of classes in the classification problem and the dataloaders
    """
    if kwargs is None:
        kwargs = {}
    generator = torch.Generator().manual_seed(seed)
    datasets = [dataset_class(root=data_path, split=split, download=download, transform=transform, **kwargs) for split
                in
                splits]
    dataloaders = [DataLoader(
        dataset, pin_memory=torch.cuda.is_available(), num_workers=len(os.sched_getaffinity(0)),
        batch_size=batch_size, shuffle=(i == 0), generator=generator)
                   for i, dataset in enumerate(datasets)]
    output_size = len(datasets[0].classes)
    return output_size, tuple(dataloaders), torch.nn.CrossEntropyLoss()


def get_dataloaders(
        batch_size: int,
        remove_idxs: Optional[Iterable[int]] = None) \
        -> List[Tuple[Optional[int],
        Tuple[DataLoader, DataLoader, DataLoader],
        torch.nn.Module]]:
    """Returns the dataloader of the large-scale continual learning experiment.

    The dataloaders for the indices in remove_idxs are ([], [], []). This can be
    useful if the loading of the prior dataset takes long or is not available.

    Args:
        batch_size: The batch size
        remove_idxs: The indices that should not load the dataloaders

    Returns:
        A tuple of output dimensions and train-, val-, and test-dataloaders
    """
    if remove_idxs is None:
        remove_idxs = []
    splits = ['train', 'val', 'test']
    wrgbd_extra_categories = ['banana', 'coffee_mug', 'stapler', 'flashlight', 'apple']

    datasets = [
                   {
                       'dataset_class': ImageNet,
                       'download': None,
                       'data_path': os.path.join(torch_data_path, 'ImageNet'),
                       'splits': ['train', 'val', 'val'],
                       'transform': Compose(
                           [
                               Resize(256),
                               CenterCrop(224),
                               ToTensor(),
                               Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                           ]),
                       'kwargs': {}
                   },
                   {
                       'dataset_class': WRGBD,
                       'download': True,
                       'data_path': torch_data_path,
                       'splits': splits,
                       'kwargs': {
                           'excluded_categories': wrgbd_extra_categories
                       },
                       'transform': Compose(
                           [
                               Resize([224, 224]),
                               ToTensor(),
                               Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                           ]),
                   },
               ] + [
                   {
                       'dataset_class': WRGBD,
                       'download': True,
                       'data_path': torch_data_path,
                       'splits': splits,
                       'kwargs': {
                           'categories': [category]
                       },
                       'transform': Compose(
                           [
                               Resize([224, 224]),
                               ToTensor(),
                               Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                           ]),
                   } for category in wrgbd_extra_categories
               ]

    dataloaders = [get_train_val_test_split_dataloaders(
        batch_size=batch_size,
        **dataset)
                   if idx not in remove_idxs else (None, ([], [], []), None)
                   for idx, dataset in enumerate(datasets)]
    return dataloaders
