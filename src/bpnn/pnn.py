"""Multiple progressive neural network variations."""
import os
from abc import abstractmethod
from copy import deepcopy
from json import dumps
from math import sqrt, prod
from typing import List, Dict, Union, Optional, Any, Tuple, Callable, Iterable, \
    Set

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .utils import fit, base_path, run_epoch, get_dataset_and_name, \
    mc_allester_bound, catoni_bound
from src.utils import evaluate_fewshot_uncertainty


def nested_module_dict(
        module_dict: nn.ModuleDict,
        name: str,
        module: nn.Module) -> nn.ModuleDict:
    """Creates a nested ModuleDict.

    The name is split by "." and each part of the split defines a new level in
    the nested dict.

    Args:
        module_dict: A ModuleDict
        name: a name containing arbitrary many "."
        module: the module that should be added in the final level

    Returns:
        The nested ModuleDict

    Examples:
        >>> name = 'test.1.test'
        >>> module = nn.Linear(10, 20)
        >>> nested_module_dict(nn.ModuleDict({}), name, module)
        ModuleDict(
            (test): ModuleDict(
                (1): ModuleDict(
                    (test): Linear(in_features=10, out_features=20, bias=True)
                    )
                )
            )
    """
    names = name.split('.')
    current_module_dict = module_dict
    for name in names[:-1]:
        new_module_dict = nn.ModuleDict({})
        current_module_dict[name] = new_module_dict
        current_module_dict = new_module_dict
    current_module_dict[names[-1]] = module
    return module_dict


class LateralConnection(nn.Module):
    """The lateral connection.

    This module aggregates and combines the records from the source layers.
    """

    def __init__(
            self,
            source_layer_names: List[str],
            forward_record: Dict[nn.Module, Tensor],
            lateral_layer_names: List[str],
            name_to_module: Callable[[str], nn.Module]):
        """LateralConnection initializer.

        Args:
            source_layer_names: The names of the source layers
            forward_record: A dict mapping modules on their input to their
                forward method
            lateral_layer_names: The names of the lateral layers
            name_to_module: A function mapping a name to its module
        """
        super().__init__()
        self.source_layer_names = source_layer_names
        self.forward_record = forward_record
        self.lateral_layer_names = lateral_layer_names
        self.name_to_module = name_to_module

    def forward(self, x):
        num_connections = len(self.source_layer_names) + 1
        x = x / num_connections
        for source_layer_name, lateral_layer_name in zip(self.source_layer_names, self.lateral_layer_names):
            source_layer = self.name_to_module(source_layer_name)
            assert source_layer in self.forward_record.keys()
            lateral_layer = self.name_to_module(lateral_layer_name)
            x = x + lateral_layer(self.forward_record[source_layer]) / num_connections
        return x


def replace_module(
        old_module_name: str,
        new_module: nn.Module,
        network: nn.Module):
    """ Replaces old_module with new_module in network.

    Args:
        old_module_name: The name of the old module
        new_module: The new module
        network: The network (should contain the old_module)
    """
    name_to_module = dict(network.named_modules())
    assert old_module_name in name_to_module.keys()
    old_modules = old_module_name.split('.')
    module_container_name, module_name = '.'.join(old_modules[:-1]), old_modules[-1]
    module_container = name_to_module[module_container_name] if module_container_name else network
    module_container.__setattr__(module_name, new_module)


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Networks (PNN).

    This network architecture adds weighted layers (lateral connections) between
    layers of previously trained and newly added columns that are jointly
    trained with the new column.

    We use forward hooks to implement PNN for arbitrary network structures.
    """

    def __init__(
            self,
            base_network: nn.Module,
            backbone: nn.Module = None,
            last_layer_name: Optional[str] = None,
            lateral_connections: Optional[List[str]] = None):
        """ProgressiveNeuralNetwork initializer.

        Args:
            base_network: A PyTorch model
            backbone: A PyTorch model that is used as a backbone for the
                Progressive Neural Network. If not provided, no backbone is used.
            last_layer_name: The name of the last layer
            lateral_connections: The names of the layers that should have
                lateral connections
        """
        super().__init__()
        self.backbone = backbone if backbone is not None else nn.Sequential()
        self.backbone.eval()
        self.base_network = base_network

        if lateral_connections is None:
            lateral_connections = []
        all_names = [name for name, _ in base_network.named_modules()]
        assert set(lateral_connections).issubset(all_names), \
            f'All lateral connections should be in the base network. ' \
            f'Given {lateral_connections} but only {all_names} are available.'

        self.lateral_connections = lateral_connections

        self.is_classification: List[bool] = []

        self.networks: nn.ModuleList[nn.ModuleList[Union[nn.Module, nn.ModuleDict]]] \
            = nn.ModuleList([])

        self.last_layer_name = last_layer_name

        self.forward_record = {}
        self.lateral_forward_pre_hooks = []
        self.name_to_module = {}
        self._update_base_network()

    @property
    def previous_tasks(self):
        """The number of previously finished tasks.

        Returns:
            The number of previously finished tasks.
        """
        return len(self.networks) + 1

    def _update_base_network(self):
        base_network_name_to_module = dict(self.base_network.named_modules())
        for lateral_connection in self.lateral_connections:
            lateral_module = base_network_name_to_module[lateral_connection]
            replace_module(
                lateral_connection,
                nn.Sequential(
                    lateral_module,
                    LateralConnection([], {}, [], None)
                ),
                self.base_network)

    def full_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the whole state.

        Returns:
            A dictionary containing the whole state
        """
        full_state_dict = {
            'last_layer_name': self.last_layer_name,
            'networks': self.networks.state_dict(),
            'base_network': self.base_network.state_dict(),
            'base_network_string': str(self.base_network),
            'backbone': self.backbone.state_dict(),
            'backbone_string': str(self.backbone),
            'lateral_connections': self.lateral_connections,
            'is_classification': self.is_classification
        }
        return full_state_dict

    def load_full_state_dict(self, full_state_dict: Dict[str, Any]):
        """Copies the whole state from full_state_dict.

        Args:
            full_state_dict: A dict containing the full state
        """
        self.lateral_connections = full_state_dict['lateral_connections']
        self.last_layer_name = full_state_dict['last_layer_name']
        if not any(isinstance(module, LateralConnection) for module in self.base_network.modules()):
            self._update_base_network()
        self.base_network.load_state_dict(full_state_dict['base_network'])
        self.backbone.load_state_dict(full_state_dict['backbone'])
        if full_state_dict['networks']:
            num_columns = max(int(key[0]) for key in full_state_dict['networks'].keys()) + 1
            for i in range(num_columns):
                ending = '.0' if self.last_layer_name in self.lateral_connections else ''
                output_size = full_state_dict['networks'][f'{i}.{i}.{self.last_layer_name}{ending}.weight'].shape[0]
                self.add_new_column(full_state_dict['is_classification'][i], output_size, False, False)
            self.networks.load_state_dict(full_state_dict['networks'])

    def train(self, mode: bool = True):
        """Sets the module in training mode.

        This means that the last column is trained.

        Args:
            mode: whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.

        Returns:
            self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.networks:
            self.networks[-1].train(mode)
        return self

    def named_modules(self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        backbone_modules = set(self.backbone.modules())
        memo = memo | backbone_modules
        return super().named_modules(memo, prefix, remove_duplicate)

    def _lateral_forward_pre_hook(self, module, input):
        self.forward_record[module] = input[0]

    def _name_to_module(self, name):
        if not self.name_to_module:
            self.name_to_module = dict(self.networks.named_modules())
        return self.name_to_module[name]

    def add_new_column(
            self,
            is_classification: bool = True,
            output_size: Optional[int] = None,
            differ_from_previous: bool = False,
            resample_base_network: bool = False):
        """Adds a new column.

        Args:
            is_classification: Whether the new column is a classification or a
                regression task.
            output_size: The dimension of the output of the last layer of the
                new column (is usually the number of classes in the
                corresponding dataset)
            differ_from_previous: Whether the new weights should be altered
                slightly
            resample_base_network: Whether the weights should be resampled
                completely
        """
        self.is_classification.append(is_classification)

        self.base_network.requires_grad_(False)
        self.networks.requires_grad_(False)
        self.networks.eval()

        intra_column = deepcopy(self.base_network)
        if output_size is not None:
            assert self.last_layer_name not in self.lateral_connections
            last_layer = dict(intra_column.named_modules())[self.last_layer_name]
            assert isinstance(last_layer, nn.Linear)
            new_last_layer = nn.Linear(last_layer.in_features, output_size, bias=last_layer.bias is not None) \
                .requires_grad_(False)
            replace_module(self.last_layer_name, new_last_layer, intra_column)

        if resample_base_network:
            intra_column.apply(
                lambda module: module.reset_parameters()
                if hasattr(module, 'reset_parameters') else None)

        inter_column = [nn.ModuleDict({}) for _ in self.networks]
        for hook in self.lateral_forward_pre_hooks:
            hook.remove()
        for lateral_connection in self.lateral_connections:
            name = f'{lateral_connection}.0'

            self.lateral_forward_pre_hooks.append(
                dict(intra_column.named_modules())[name].register_forward_pre_hook(
                    self._lateral_forward_pre_hook))

            for inter_column_connections, column_previous in zip(inter_column, self.networks):
                column_previous_name_to_module = dict(column_previous[-1].named_modules())
                new_module = deepcopy(column_previous_name_to_module[name])

                self.lateral_forward_pre_hooks.append(
                    column_previous_name_to_module[name].register_forward_pre_hook(
                        self._lateral_forward_pre_hook))

                nested_module_dict(inter_column_connections, name, new_module)
            replace_module(
                f'{lateral_connection}.1',
                LateralConnection(
                    [f'{i}.{i}.{lateral_connection}.0' for i in range(len(self.networks))],
                    self.forward_record,
                    [f'{len(self.networks)}.{i}.{lateral_connection}.0' for i in range(len(self.networks))],
                    self._name_to_module),
                intra_column)

        new_network = nn.ModuleList([*inter_column, intra_column])

        if differ_from_previous:
            for param in new_network.parameters():
                param += (torch.randn_like(param) / sqrt(prod(param.shape))) \
                         * torch.linalg.norm(param) * torch.finfo(param.dtype).eps

        self.networks.append(new_network.requires_grad_(True))
        self.name_to_module = dict(self.networks.named_modules())
        torch.cuda.empty_cache()

    def forward(self, x: Tensor):
        assert self.networks, 'no column is available, please call add_new_column before forward_once'
        x = self.backbone(x)
        out = [column[-1](x) for column in self.networks]
        self.forward_record.clear()
        return out


class ProbabilisticProgressiveNeuralNetwork(ProgressiveNeuralNetwork):
    """Base class for Probabilistic Progressive Neural Networks (PPNN).

    In addition to the structure of Progressive Neural Networks, it allows
    sampling weights during evaluation differently that during training.
    Additionally, it implements Bayesian model averaging in the forward method.
    """

    def __init__(
            self,
            base_network: nn.Module,
            backbone: nn.Module = None,
            last_layer_name: Optional[str] = None,
            lateral_connections: Optional[List[str]] = None,
            train_resample_slice: slice = slice(None),
            train_num_samples: int = 1,
            eval_resample_slice: slice = slice(None),
            eval_num_samples: int = 100):
        """ProbabilisticProgressiveNeuralNetwork initializer.

        Args:
            base_network: A PyTorch model
            backbone: A PyTorch model that is used as a backbone for the
                Progressive Neural Network. If not provided, no backbone is used.
            last_layer_name: The name of the last layer
            lateral_connections: The names of the layers that should have
                lateral connections
            train_resample_slice: The columns from which the weight should be
                sampled during training
            train_num_samples: The number of samples used for each forward pass
                during training
            eval_resample_slice: The columns from which the weight should be
                sampled during evaluation
            eval_num_samples: The number of samples used for each forward pass
                during evaluation
        """
        super().__init__(
            base_network, backbone, last_layer_name,
            lateral_connections)

        self.train_resample_slice = train_resample_slice
        self.train_num_samples = train_num_samples

        self.eval_resample_slice = eval_resample_slice
        self.eval_num_samples = eval_num_samples

    def full_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the whole state.

        Returns:
            A dictionary containing the whole state
        """
        full_state_dict = super().full_state_dict()
        full_state_dict['train_resample_slice'] = self.train_resample_slice
        full_state_dict['train_num_samples'] = self.train_num_samples

        full_state_dict['eval_resample_slice'] = self.eval_resample_slice
        full_state_dict['eval_num_samples'] = self.eval_num_samples
        return full_state_dict

    def load_full_state_dict(self, full_state_dict: Dict[str, Any]):
        """Copies the whole state from full_state_dict.

        Args:
            full_state_dict: A dict containing the full state
        """
        super().load_full_state_dict(full_state_dict)

        self.train_resample_slice = full_state_dict['train_resample_slice']
        self.train_num_samples = full_state_dict['train_num_samples']

        self.eval_resample_slice = full_state_dict['eval_resample_slice']
        self.eval_num_samples = full_state_dict['eval_num_samples']

    @abstractmethod
    def _sample_and_replace(
            self,
            resample_slice: slice):
        raise NotImplementedError

    def sample_and_replace(
            self,
            resample_slice: Optional[slice] = None):
        """Samples and replaces the weights from all columns in resample_slice.

        Args:
            resample_slice: The indices of the columns that should be sampled
        """
        if resample_slice is None:
            resample_slice = self.train_resample_slice if self.training else self.eval_resample_slice
        self._sample_and_replace(resample_slice)

    def forward_once(
            self,
            x: Tensor,
            resample_slice: Optional[slice] = None):
        self.sample_and_replace(resample_slice=resample_slice)
        return super().forward(x)

    def forward(
            self,
            x: Tensor,
            num_samples: Optional[int] = None,
            resample_slice: Optional[slice] = None):
        assert self.networks, \
            'no column is available, please call add_new_column before forward'
        if num_samples is None:
            num_samples = self.train_num_samples if self.training else self.eval_num_samples
        assert num_samples >= 1, \
            f'num_samples should be larger or equal than 1, but it is {num_samples}'

        logits = [self.forward_once(x, resample_slice=resample_slice)
                  for _ in range(num_samples)]
        out = [torch.stack([l[j] for l in logits], dim=1)
               for j in range(len(self.networks))]
        return out


class DropoutProgressiveNeuralNetwork(ProbabilisticProgressiveNeuralNetwork):
    """Dropout Progressive Neural Network (DPNN).

    This is a Probabilistic Progressive Neural Network that samples different
    weights with dropout.
    """

    def __init__(
            self,
            base_network: nn.Module,
            backbone: nn.Module = None,
            last_layer_name: Optional[str] = None,
            lateral_connections: Optional[List[str]] = None,
            train_resample_slice: slice = slice(None),
            train_num_samples: int = 1,
            eval_resample_slice: slice = slice(None),
            eval_num_samples: int = 100,
            dropout_probability: float = 0.5,
            dropout_positions: Optional[List[str]] = None):
        """DropoutProgressiveNeuralNetwork initializer.

        Args:
            base_network: A PyTorch model
            backbone: A PyTorch model that is used as a backbone for the
                Progressive Neural Network. If not provided, no backbone is used.
            last_layer_name: The name of the last layer
            lateral_connections: The names of the layers that should have
                lateral connections
            train_resample_slice: The columns from which the weight should be
                sampled during training
            train_num_samples: The number of samples used for each forward pass
                during training
            eval_resample_slice: The columns from which the weight should be
                sampled during evaluation
            eval_num_samples: The number of samples used for each forward pass
                during evaluation
            dropout_probability: The probability of the dropout layers
            dropout_positions: The layer names where a dropout layer should be
                appended
        """

        if dropout_positions is None:
            dropout_positions = deepcopy(lateral_connections)

        base_network_name_to_module = dict(base_network.named_modules())
        for dropout_position in dropout_positions:
            if dropout_position in lateral_connections:
                lateral_connections[lateral_connections.index(dropout_position)] = f'{dropout_position}.0'
            module = base_network_name_to_module[dropout_position]
            replace_module(
                dropout_position,
                nn.Sequential(
                    module,
                    nn.Dropout(dropout_probability)
                ),
                base_network)

        super().__init__(
            base_network, backbone, last_layer_name, lateral_connections,
            train_resample_slice, train_num_samples, eval_resample_slice, eval_num_samples)

    @staticmethod
    def _activate_dropout(module):
        if type(module) == nn.Dropout:
            module.train()

    @staticmethod
    def _deactivate_dropout(module):
        if type(module) == nn.Dropout:
            module.eval()

    def _sample_and_replace(
            self,
            resample_slice: slice):
        self.networks.apply(self._deactivate_dropout)
        for column in self.networks[resample_slice]:
            column.apply(self._activate_dropout)


def dataset_step_pnn(
        model: ProgressiveNeuralNetwork,
        dataloader: Tuple[DataLoader, DataLoader, DataLoader],
        loss_function: nn.Module,
        output_size: int,
        weight_decay: float,
        learning_rate: float,
        num_epochs: int,
        patience: int,
        **kwargs) -> Dict[str, List[Dict[str, float]]]:
    """The dataset step of PNN for fit_pnn.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloader: A tuple of train-, val-, and test-dataloaders
        loss_function: The loss function
        output_size: The dimension of the network output
        weight_decay: The weight decay used in the BPNN
        learning_rate: The learning rate used to train the column
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped

    Returns:
        The metrics computed during the training
    """
    is_classification = type(loss_function) is nn.CrossEntropyLoss
    model.add_new_column(is_classification=is_classification, output_size=output_size)
    train_metrics = fit(
        model, dataloader, loss_function,
        weight_decay=weight_decay,
        is_classification=model.is_classification,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience)
    return train_metrics


def dataset_step_ppnn(
        model: ProbabilisticProgressiveNeuralNetwork,
        dataloader: Tuple[DataLoader, DataLoader, DataLoader],
        loss_function: nn.Module,
        output_size: int,
        weight_decay: float,
        learning_rate: float,
        num_epochs: int,
        patience: int,
        **kwargs) -> Dict[str, List[Dict[str, float]]]:
    """The dataset step of PPNN for fit_pnn.

    Args:
        model: A BayesianProgressiveNeuralNetwork object
        dataloader: A tuple of train-, val-, and test-dataloaders
        loss_function: The loss function
        output_size: The dimension of the network output
        weight_decay: The weight decay used in the BPNN
        learning_rate: The learning rate used to train the column
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped

    Returns:
        The metrics computed during the training
    """
    is_classification = type(loss_function) is nn.CrossEntropyLoss
    model.add_new_column(
        is_classification=is_classification,
        output_size=output_size)
    eval_num_samples = model.eval_num_samples
    model.eval_num_samples = 1
    train_metrics = fit(
        model, dataloader, loss_function,
        weight_decay=weight_decay,
        is_classification=model.is_classification,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience)
    model.eval_num_samples = eval_num_samples
    return train_metrics


def fit_pnn(
        model: ProgressiveNeuralNetwork,
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], nn.Module]],
        dataset_step: Callable,
        weight_decay: float = 1e-5,
        learning_rate: float = 2e-3,
        num_epochs: int = 100,
        patience: int = 10,
        name: Optional[str] = None,
        eval_every_task: bool = True,
        compute_pac_bounds: bool = True,
        confidence: float = 0.8,
        **kwargs) -> List[Dict]:
    """Runs the full optimization routine for ProgressiveNeuralNetworks.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: A tuple of output dimensions, train-, val-, and test-
            dataloaders and loss function
        dataset_step: A function that represents adding and training a new column
        weight_decay: The weight decay used in the PNN
        learning_rate: The learning rate used in the full training
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        name: The name of the run (also the save path)
        eval_every_task: Whether the model should be evaluated on each column
            after training a new column
        compute_pac_bounds: Whether PAC-Bayesian bounds should be computed
            (needs that model has the method kl_divergence)
        confidence: The confidence used for the PAC-Bayesian bounds

    Returns:
        A dict containing the training and evaluation metrics
    """
    metrics = []
    model_path = os.path.join(base_path, 'models', name)
    if name is not None and not os.path.exists(model_path):
        os.makedirs(model_path)

    if name is not None:
        torch.save(model.full_state_dict(), os.path.join(model_path, 'full_state_dict_-1.pt'))

    previous_tasks = model.previous_tasks  # + 1 for prior task
    for task, (output_size, dataloader, loss_function) in enumerate(dataloaders[previous_tasks:], start=previous_tasks):
        if not isinstance(loss_function, (nn.CrossEntropyLoss, nn.MSELoss)):
            raise ValueError('Only CrossEntropyLoss and MSELoss are currently supported')
        torch.cuda.empty_cache()
        train_dataset, dataset_name = get_dataset_and_name(dataloader[0])
        print(f'Task {task}: {dataset_name}')

        train_metrics = dataset_step(
            model, dataloader, loss_function, output_size, weight_decay,
            learning_rate, train_dataset=train_dataset,
            num_epochs=num_epochs, patience=patience, **kwargs)

        if name is not None:
            torch.save(model.full_state_dict(), os.path.join(model_path, f'full_state_dict_{task}.pt'))

        # evaluate model
        first_task = 1 if eval_every_task else task
        is_classification = [type(loss_function) is type(nn.CrossEntropyLoss())
                             for (_, _, loss_function), _ in zip(dataloaders[1:], range(task))]
        metrics.append(
            {
                'train': train_metrics,
                'test': evaluate_pnn(
                    model, dataloaders[first_task:task + 1],
                    range(first_task - 1, task),
                    compute_pac_bounds=compute_pac_bounds,
                    confidence=confidence,
                    is_classification=is_classification)
            })
    return metrics


def evaluate_pnn(
        model: ProgressiveNeuralNetwork,
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], nn.Module]],
        dims: Iterable[int],
        metrics: Union[str, List[str]] = 'all',
        compute_pac_bounds: bool = True,
        confidence: float = 0.8,
        is_classification: Union[bool, List[bool]] = True):
    """Evaluates ProgressiveNeuralNetworks.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: A tuple of output dimensions, train-, val-, and
            test-dataloaders and loss function
        dims: The columns that should be used for each dataloader
        metrics: 'all' or a list of metrics (see bpnn.utils.MetricsSet)
        compute_pac_bounds: Whether PAC-Bayesian bounds should be computed
            (needs that model has the method kl_divergence)
        confidence: The confidence used for the PAC-Bayesian bounds
        is_classification: Whether the task is a classification task

    Returns:
        A dict containing the evaluation metrics
    """
    model.eval()
    model.requires_grad_(False)
    test_metrics = []
    for local_dim, (_, local_dataloader, loss_function) in zip(dims, dataloaders):
        test_metric = run_epoch(
            model, local_dataloader[-1], loss_function,
            metrics=metrics, metrics_dim=local_dim,
            is_classification=is_classification)
        if local_dim == list(dims)[-1] and compute_pac_bounds and is_classification[local_dim]:
            train_metric = run_epoch(
                model, local_dataloader[0], loss_function,
                metrics='accuracy', metrics_dim=local_dim,
                is_classification=is_classification)
            test_metric['train accuracy'] = train_metric['accuracy']
            expected_empirical_risk = 1 - train_metric['accuracy'] / 100
            kl_divergence = model.kl_divergence(penalty_slice=slice(local_dim, local_dim + 1))
            test_metric['KL divergence'] = kl_divergence.item()
            len_data = len(local_dataloader[0].dataset)
            test_metric['McAllester bound'] = mc_allester_bound(
                expected_empirical_risk,
                kl_divergence,
                len_data,
                confidence=confidence).item()
            test_metric['Catoni bound'] = catoni_bound(
                expected_empirical_risk,
                kl_divergence,
                len_data,
                confidence=confidence).item()

        local_dataset, local_dataset_name = get_dataset_and_name(local_dataloader[-1])

        test_metric['dataset'] = local_dataset_name
        print(
            f'{local_dataset_name}:\n'
            f'{dumps(test_metric, sort_keys=True, indent=4, default=str)}')
        test_metrics.append(test_metric)

    return test_metrics

def evaluate_fewshot(model, dataloaders, device, metrics_dim=0):
    """Evaluates ProgressiveNeuralNetworks for few shot classification.
    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: test data loader
        device: cpu or gpu
    Returns:
        A dict containing the evaluation metrics
    """
    # model preparations
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    # collect results
    target_list = []
    prob_list = []
    for iter_num, (features, targets) in enumerate(dataloaders):
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        probs = F.softmax(logits[metrics_dim], dim=-1)
        probs = probs.mean(dim=1)        
        target_list.append(targets.cpu().numpy())
        prob_list.append(probs.cpu().numpy())
    prob_numpy = np.concatenate(prob_list, axis=0)
    target_numpy = np.concatenate(target_list, axis=0)
    test_metrics = evaluate_fewshot_uncertainty(prob_numpy, target_numpy)
    return test_metrics