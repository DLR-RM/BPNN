"""Dataloader fewshot learning. Read to get torch dataloader"""
import numpy as np
import torch
import os

from typing import List, Tuple, Optional, Dict, Iterable
from torchvision.datasets import ImageNet
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
from src.bpnn.utils import seed, torch_data_path


def tensor_to_torch_loader(tensor_x: torch.Tensor, tensor_y: torch.Tensor, is_train: bool = False, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """Conversion of tensors into torch dataset loaders."""
    dataset = TensorDataset(tensor_x, tensor_y)
    if is_train:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def train_torch_loader(dataset: str, train_split: str, shot: int, dataload_dir: str, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """Function to get few shot train torch data loader."""
    # load the numpy arrays
    filename_train_x = dataload_dir + './' + dataset + '_' + train_split + '_' + str(shot) + '_' + 'input.npy'
    filename_train_y = dataload_dir + './' + dataset + '_' + train_split + '_' + str(shot) + '_' + 'label.npy'
    input_x = np.load(file=filename_train_x)
    input_y = np.load(file=filename_train_y)
    
    # convert them into torch data
    tensor_x = torch.Tensor(input_x).permute(0, 3, 1, 2)
    tensor_y = torch.Tensor(input_y).type(torch.LongTensor)

    # convert them into torch dataloader
    dataloader = tensor_to_torch_loader(tensor_x, tensor_y, is_train=True, batch_size=batch_size)

    return dataloader

def test_torch_loader(dataset: str, test_split: str, dataload_dir: str, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """Function to get few shot test torch data loader."""
    # load the numpy arrays
    filename_test_x = dataload_dir + './' + dataset + '_' + test_split + '_' + 'input.npy'
    filename_test_y = dataload_dir + './' + dataset + '_' + test_split + '_' + 'label.npy'
    input_x = np.load(file=filename_test_x)
    input_y = np.load(file=filename_test_y)

    # convert them into torch data
    tensor_x = torch.Tensor(input_x).permute(0, 3, 1, 2)
    tensor_y = torch.Tensor(input_y).type(torch.LongTensor)
    
    # convert them into torch dataloader
    dataloader = tensor_to_torch_loader(tensor_x, tensor_y, is_train=False, batch_size=batch_size)

    return dataloader

def validation_torch_loader(dataset: str, shot: int, dataload_dir: str, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """Function to get few shot test torch data loader."""
    # load the numpy arrays
    filename_val_x = dataload_dir + './' + dataset + '_validation_' + str(shot) + '_' + 'input.npy'
    filename_val_y = dataload_dir + './' + dataset + '_validation_' + str(shot) + '_' + 'label.npy'
    input_x = np.load(file=filename_val_x)
    input_y = np.load(file=filename_val_y)

    # convert them into torch data
    tensor_x = torch.Tensor(input_x).permute(0, 3, 1, 2)
    tensor_y = torch.Tensor(input_y).type(torch.LongTensor)
    
    # convert them into torch dataloader
    dataloader = tensor_to_torch_loader(tensor_x, tensor_y, is_train=False, batch_size=batch_size)

    return dataloader

def get_train_val_test_split_dataloaders(dataset_class: type,
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
    dataloaders = [DataLoader(dataset, pin_memory=torch.cuda.is_available(), num_workers=len(os.sched_getaffinity(0)),
                              batch_size=batch_size, shuffle=(i == 0), generator=generator)
                   for i, dataset in enumerate(datasets)]
    output_size = len(datasets[0].classes)
    return output_size, tuple(dataloaders), torch.nn.CrossEntropyLoss()