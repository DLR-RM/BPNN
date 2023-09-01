"""Script to run and evaluate K-FOC experiments."""
import json
import os
from time import time
from typing import Union, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.curvature.curvatures import BlockDiagonal, Diagonal, KFAC, KFOC
from src.curvature.lenet5 import lenet5
from src.bpnn.utils import seed, base_path, compute_curvature, seed_all_rng, \
    fit, torch_data_path

raw_data_path = os.path.join(base_path, 'data', 'raw')


def get_fisher(state: Union[torch.Tensor, List[torch.Tensor]],
               dtype: torch.dtype = torch.float) -> torch.Tensor:
    """Returns the Fisher Information Matrix (FIM) from an approximation.

    Args:
        state: The state of a curvature object
        dtype: The PyTorch data type of the resulting matrix

    Returns:
        A tensor of the FIM
    """
    assert isinstance(state, torch.Tensor) or isinstance(state, list)
    if isinstance(state, torch.Tensor):
        assert state.ndim in [1, 2]
        state = state.to(dtype)
        if state.ndim == 1:
            return torch.diag(state)
        elif state.ndim == 2:
            if state.shape[0] == state.shape[1]:
                return state
            else:
                return torch.diag(state.view(-1))
    elif isinstance(state, list):
        return torch.kron(state[1].to(dtype), state[0].to(dtype))


def relative_difference(ground_truth: torch.Tensor,
                        other: torch.tensor,
                        ord: Optional[str] = None,
                        dim: Optional[int] = None) -> torch.Tensor:
    """Computes the relative difference of the two tensors."""
    return torch.linalg.norm(ground_truth - other, ord=ord, dim=dim) \
           / (torch.linalg.norm(ground_truth, ord=ord, dim=dim) + torch.finfo(ground_truth.dtype).eps)


def evaluate_curvatures(network: nn.Module,
                        dataset: Dataset,
                        dataset_name: str,
                        results: Dict,
                        runtime_dict: Dict,
                        batch_sizes: List[int],
                        seeds: List[int],
                        categorical: bool = False,
                        layer_types: Union[List[str], str] = 'Linear'):
    """Computes and evaluates the curvatures w.r.t. approximation quality and
    runtime.

    Args:
        network: A PyTorch model
        dataset: A PyTorch dataset
        dataset_name: The name of the dataset
        results: The result dict
        runtime_dict: The runtime dict
        batch_sizes: The batch sizes to compute the curvature
        seeds: The random seeds that are used
        categorical: Whether the dataset is categorical or continuous
        layer_types: Types of layers for which to compute the curvature
    """
    for batch_size in batch_sizes:
        for seed in seeds:
            dataloader = DataLoader(dataset, pin_memory=False, num_workers=0,
                                    batch_size=batch_size, shuffle=True,
                                    generator=torch.Generator().manual_seed(seed))
            curv_dict = {
                'BlockDiagonal': BlockDiagonal(network, layer_types=layer_types),
                'Diagonal': Diagonal(network, layer_types=layer_types),
                'K-FAC': KFAC(network, layer_types=layer_types),
                'K-FOC_approx': KFOC(network, approx=True, layer_types=layer_types),
                'K-FOC_running': KFOC(network, approx=False, layer_types=layer_types),
                'no curvature': None
            }
            for name, curv in curv_dict.items():
                start_time = time()
                curvs = [] if name == 'no curvature' else [curv]
                compute_curvature(network, curvs, dataloader,
                                  return_data_log_likelihood=False,
                                  categorical=categorical, seed=seed)
                end_time = time()
                set_dict(runtime_dict, [dataset_name, name, batch_size, seed], end_time - start_time)
            for layer, bd_state in curv_dict['BlockDiagonal'].state.items():
                for approx_name, approx in curv_dict.items():
                    if approx_name == 'BlockDiagonal' or approx_name == 'no curvature':
                        continue
                    approx_state = get_fisher(approx.state[layer])
                    rel_error = relative_difference(bd_state, approx_state)
                    set_dict(results, [dataset_name, str(layer), approx_name, batch_size, seed], rel_error.item())


def set_dict(d: Dict,
             keys: List,
             value: Any):
    """Creates a nested structures with given keys and assigns a value."""
    if len(keys) == 1:
        d[keys[0]] = value
        return
    if keys[0] not in d.keys():
        d[keys[0]] = {}
    set_dict(d[keys[0]], keys[1:], value=value)


def standardize_tensor(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Subtracts the mean and divides the standard deviation in a dimension."""
    return (tensor - tensor.mean(dim=dim, keepdim=True)) / tensor.std(dim=dim, keepdim=True)


def get_boston_housing_dataset(standardize: bool = True):
    """Returns the Boston Housing Dataset"""
    bos = load_boston()
    features = torch.tensor(bos.data, dtype=torch.float)
    if standardize:
        features = standardize_tensor(features)
    targets = torch.tensor(bos.target, dtype=torch.float)[:, None]
    return TensorDataset(features, targets)


def get_pandas_dataset(df: pd.DataFrame,
                       target_names: List[str],
                       standardize: bool = True):
    """Creates a dataset from a dataframe."""
    targets = torch.tensor(df[target_names].to_numpy(), dtype=torch.float)
    features = torch.tensor(df.drop(target_names, axis=1).to_numpy(), dtype=torch.float)
    if standardize:
        features = standardize_tensor(features)
    return TensorDataset(features, targets)


def get_concrete_compression_strength_dataset(standardize: bool = True):
    """Returns the Concrete Compression Strength Dataset"""
    df = pd.read_excel(os.path.join(raw_data_path, 'Concrete_Data.xls'))
    target_names = ['Concrete compressive strength(MPa, megapascals) ']
    return get_pandas_dataset(df, target_names, standardize)


def get_energy_efficiency_dataset(standardize: bool = True):
    """Returns the Energy Efficiency Dataset"""
    df = pd.read_excel(os.path.join(raw_data_path, 'ENB2012_data.xlsx'))
    target_names = ['Y1', 'Y2']
    return get_pandas_dataset(df, target_names, standardize)


def plot_results(results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[int, float]]]]],
                 target_path: str):
    """Averages over the seeds and plots the batch size against the relative
    error for each dataset and layer."""
    for dataset_name, layers in results.items():
        for layer_idx, (layer_name, approx) in enumerate(layers.items()):
            for approx_name, batch_size_dict in approx.items():
                batch_sizes, values = zip(*[(batch_size, [value for value in seeds.values()])
                                            for batch_size, seeds in batch_size_dict.items()])
                batch_sizes = np.array([int(batch_size) for batch_size in batch_sizes])
                values = np.array(values)
                plt.gca().set_title(layer_name)
                plt.plot(batch_sizes, values.mean(axis=1), label=approx_name)
                plt.gca().fill_between(batch_sizes,
                                       values.min(axis=1),
                                       values.max(axis=1),
                                       alpha=.3)
            plt.semilogx()
            plt.xlabel('batch size')
            plt.ylabel('relative error')
            plt.grid()
            plt.legend()
            print(layer_idx, layer_name)
            plt.savefig(os.path.join(os.path.join(target_path, f'{dataset_name}_{layer_idx}.png')))


if __name__ == '__main__':
    results = {}
    runtime_dict = {}
    seeds = [seed_all_rng() for _ in range(10)]

    for dataset_name, dataset in {'Boston Housing': get_boston_housing_dataset(),
                                  'Concrete Compression Strength': get_concrete_compression_strength_dataset(),
                                  'Energy Efficiency': get_energy_efficiency_dataset()}.items():
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, pin_memory=False, num_workers=0,
                                      batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
        test_dataloader = DataLoader(test_dataset, pin_memory=False, num_workers=0,
                                     batch_size=len(dataset), shuffle=False,
                                     generator=torch.Generator().manual_seed(seed))

        in_features = dataset.tensors[0].shape[1]
        out_features = dataset.tensors[1].shape[1]

        network = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, out_features)
        )
        fit(network, (train_dataloader, None, test_dataloader), nn.MSELoss(),
            weight_decay=1e-5, is_classification=False, learning_rate=2e-3,
            use_validation_set=False, num_epochs=1_000, patience=20, metrics_run_epoch=[])

        batch_sizes = [1, 10, 100, len(dataset)]
        evaluate_curvatures(network, dataset, dataset_name, results,
                            runtime_dict, batch_sizes, seeds,
                            categorical=False, layer_types='Linear')

    network = lenet5(True)
    dataset = MNIST(torch_data_path, transform=ToTensor())
    dataset_name = 'MNIST'
    batch_sizes = [1, 10, 100, 1000]
    evaluate_curvatures(network, dataset, dataset_name, results,
                        runtime_dict, batch_sizes, seeds,
                        categorical=True, layer_types='Conv2d')

    print('Results:')
    print(json.dumps(results, indent=4, sort_keys=True))
    with open(os.path.join(base_path, 'results', 'kfoc', 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print('Runtime:')
    print(json.dumps({
        dataset_name: {
            name: np.mean([el for batches in approx_runtime.values() for el in batches.values()]) for
            name, approx_runtime in dataset_runtime.items()
        } for dataset_name, dataset_runtime in runtime_dict.items()
    }, indent=4, sort_keys=True))
    with open(os.path.join(base_path, 'results', 'kfoc', 'runtime.json'), 'w') as f:
        json.dump(runtime_dict, f, indent=4)

    target_path = os.path.join(base_path, 'reports', 'kfoc')
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    plot_results(results, target_path)

    runtime_dfs = {key: pd.DataFrame(value) for key, value in runtime_dict.items()}


    def _format_entries(d: Dict):
        values = list(d.values())
        return f'{1e3 * np.mean(values):.1f}±{1e3 * np.std(values):.1f}'


    out = pd.concat({k: v.applymap(_format_entries).T for k, v in runtime_dfs.items()}, axis=0)
    out.columns = out.columns.astype(int)
    out.to_csv(os.path.join(target_path, 'runtime.csv'))
