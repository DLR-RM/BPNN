"""Dataloader of the small-scale continual learning experiment"""
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import KMNIST, MNIST, FashionMNIST

from src.bpnn.utils import seed, torch_data_path
from tools.not_mnist import NotMNIST


def get_train_val_test_split_dataloaders(
        dataset_class: type,
        torch_data_path: str,
        split: List,
        transform,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = None) \
        -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns the training, validation and test split for a dataset.

    Args:
        dataset_class: A dataset class
        torch_data_path: The path to the torch datasets
        split: A list of [training split, validation split] for the split
            parameter of the dataset class
        transform: A torchvision transform
        batch_size: The batch size for the dataloader
        num_workers: The number of workers for the dataloader
        pin_memory: Whether to pin the memory of the dataloader

    Returns:
        The train-, val- and test-dataloader
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    train_val = dataset_class(torch_data_path, split[0], transform=transform, download=True)
    train_size = int(0.9 * len(train_val))
    val_size = len(train_val) - train_size
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(
        train_val, [train_size, val_size],
        generator=generator)
    dataloader_train = torch.utils.data.DataLoader(
        train, pin_memory=pin_memory, num_workers=num_workers,
        batch_size=batch_size, shuffle=True, generator=generator)
    dataloader_val = torch.utils.data.DataLoader(
        val, pin_memory=pin_memory, num_workers=num_workers,
        batch_size=batch_size, shuffle=False, generator=generator)
    test = dataset_class(torch_data_path, split[1], transform=transform, download=True)
    test_size = len(test)
    dataloader_test = torch.utils.data.DataLoader(
        test, pin_memory=pin_memory, num_workers=num_workers,
        batch_size=batch_size, shuffle=False, generator=generator)
    print(f'{dataset_class.__name__}: Train: {train_size}, Val: {val_size}, Test: {test_size}')
    return dataloader_train, dataloader_val, dataloader_test


def get_dataloaders(batch_size: int) \
        -> List[Tuple[Optional[int],
        Tuple[DataLoader, DataLoader, DataLoader],
        torch.nn.Module]]:
    """Returns the dataloader of the small-scale continual learning experiment.

    Args:
        batch_size: The batch size

    Returns:
        A tuple of output dimensions and train-, val-, and test-dataloaders
    """

    def to_3_channel_grayscale(x):
        """Repeats the first channel to obtain a 3-channel grayscale image."""
        return x.repeat(3, 1, 1)

    datasets = [
        {
            'dataset_class': MNIST,
            'split': [True, False],
            'transform': transforms.Compose(
                [
                    transforms.ToTensor(),
                ])
        },
        {
            'dataset_class': NotMNIST,
            'split': [True, False],
            'transform': transforms.Compose(
                [
                    transforms.ToTensor(),
                ])
        },
        {
            'dataset_class': KMNIST,
            'split': [True, False],
            'transform': transforms.Compose(
                [
                    transforms.ToTensor(),
                ])
        },
        {
            'dataset_class': FashionMNIST,
            'split': [True, False],
            'transform': transforms.Compose(
                [
                    transforms.ToTensor(),
                ])
        },
    ]

    dataloaders = [(10, get_train_val_test_split_dataloaders(
        torch_data_path=torch_data_path,
        batch_size=batch_size,
        **dataset),
                    torch.nn.CrossEntropyLoss())
                   for dataset in datasets]
    return dataloaders
