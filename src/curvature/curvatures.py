"""Various Fisher information matrix approximations."""
from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from math import sqrt
from typing import Union, List, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from numpy.linalg import cholesky
from torch import Tensor
from torch.distributions.constraints import positive_definite
from torch.nn import Module, Sequential
from tqdm import tqdm

from .utils import power_method_sum_kronecker_products_rank_1, check_and_make_pd, \
    sum_kronecker_products, power_method_sum_kronecker_products_full_rank, \
    invert_and_cholesky


class Curvature(ABC):
    """Base class for all src approximations.

    All src approximations are computed layer-wise (i.e. layer-wise independence is assumed s.t. no
    covariances between layers are computed, aka block-wise approximation) and stored in `state`.

    The src of the loss function is the matrix of 2nd-order derivatives of the loss w.r.t. the networks weights
    (i.e. the expected Hessian). It can be approximated by the expected Fisher information matrix and, under exponential
    family loss functions (like mean-squared error and cross-entropy loss) and piecewise linear activation functions
    (i.e. ReLU), becomes identical to the Fisher.

    Note:
        The aforementioned identity does not hold for the empirical Fisher, where the expectation is computed w.r.t.
        the data distribution instead of the models' output distribution. Also, because the latter is usually unknown,
        it is approximated through Monte Carlo integration using samples from a categorical distribution, initialized by
        the models' output.

    Source: `Optimizing Neural Networks with Kronecker-factored Approximate Curvature
    <https://arxiv.org/abs/1503.05671>`_
    """

    def __init__(self,
                 model: Union[Module, Sequential],
                 layer_types: Union[List[str], str] = None,
                 device: Optional[torch.device] = None):
        """Curvature class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            layer_types: Types of layers for which to compute src information. Supported are `Linear`, `Conv2d.
                If `None`, all supported types are considered. Default: None.
            device: device on which the curvature should be computed
        """
        self.model = model
        self.model_state = copy.deepcopy(model.state_dict())
        self.layer_types = list()
        if isinstance(layer_types, str):
            self.layer_types.append(layer_types)
        elif isinstance(layer_types, list):
            if layer_types:
                self.layer_types.extend(layer_types)
            else:
                self.layer_types.extend(['Linear', 'Conv2d'])
        elif layer_types is None:
            self.layer_types.extend(['Linear', 'Conv2d'])
        else:
            raise TypeError
        for _type in self.layer_types:
            assert _type in ['Linear', 'Conv2d']
        self.state = dict()
        self.inv_state = dict()

        self.hooks = list()
        self.record = dict()

        for layer in model.modules():
            if layer._get_name() in self.layer_types:
                self.record[layer] = [None, None]
                self.hooks.append(layer.register_forward_pre_hook(self._save_input))
                self.hooks.append(layer.register_full_backward_hook(self._save_output))

        self.device = device

    def to(self, device: torch.device):
        """Moves the curvature to a different device.

        Args:
            device: Device to which the curvature should be moved
        """

        def recursive_to(state):
            if isinstance(state, Tensor):
                return state.to(device)
            elif isinstance(state, list):
                return [recursive_to(s) for s in state]

        for module, state in self.state.items():
            self.state[module] = recursive_to(state)
        for module, state in self.inv_state.items():
            self.inv_state[module] = recursive_to(state)

    def remove_hooks_and_records(self):
        """Removes all hooks and records.

        Can be used after the last update step was computed to free up memory.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()
        self.record = dict()

    def _save_input(self, module, input):
        record = input[0].detach()
        if self.device:
            record = record.to(self.device)
        self.record[module][0] = record

    def _save_output(self, module, grad_input, grad_output):
        grad = grad_output[0].detach()
        if self.device:
            grad = grad.to(self.device)
        self.record[module][1] = grad * grad.size(0)

    @staticmethod
    def _replace(sample: Tensor,
                 weight: Tensor,
                 bias: Tensor = None):
        """Modifies current model parameters by adding/subtracting quantity given in `sample`.

        Args:
            sample: Sampled offset from the mean dictated by the inverse src (variance).
            weight: The weights of one model layer.
            bias: The bias of one model layer. Optional.
        """
        if bias is not None:
            bias_sample = sample[:, -1].contiguous().view(*bias.shape)
            bias.data.add_(bias_sample.to(bias.device))
            sample = sample[:, :-1]
        weight.data.add_(sample.contiguous().view(*weight.shape).to(weight.device))

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Updates the state.

        Abstract method to be implemented by each derived class individually."""
        raise NotImplementedError

    @staticmethod
    def _transform_forward(layer, forward):
        module_class = layer._get_name()
        if module_class == 'Conv2d':
            forward = F.unfold(forward, layer.kernel_size, padding=layer.padding, stride=layer.stride)

            forward = forward.data.permute(0, 2, 1)
            if layer.bias is not None:
                ones = torch.ones_like(forward[:, :, :1])
                forward = torch.cat([forward, ones], dim=2)
        else:
            forward = forward.data
            if layer.bias is not None:
                ones = torch.ones_like(forward[:, :1])
                forward = torch.cat([forward, ones], dim=1)
        return forward

    def _get_grads(self, layer):
        forward, backward = self.record[layer]
        module_class = layer._get_name()

        forward = self._transform_forward(layer, forward)

        if module_class == 'Conv2d':
            backward = backward.data.view(*backward.shape[:-2], -1)
            grads = backward @ forward

        else:
            backward = backward.data
            grads = backward[:, :, None] @ forward[:, None, :]
        return grads

    @abstractmethod
    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Inverts state.

        Abstract method to be implemented by each derived class individually.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self,
               layer: Module) -> Tensor:
        """Samples from inverted state.

        Abstract method to be implemented by each derived class individually.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        raise NotImplementedError

    def sample_and_replace(self, temperature_scaling: Union[float, Dict[Module, float]] = 1.):
        """Samples new model parameters and replaces old ones for selected
        layers, skipping all others.

        Args:
            temperature_scaling: temperature scaling :math:`\\tau`
        """
        self.model.load_state_dict(self.model_state)
        for layer in self.model.modules():
            if layer._get_name() in self.layer_types:
                temperature = temperature_scaling if isinstance(temperature_scaling, float) \
                    else temperature_scaling[layer]
                if layer._get_name() in ['Linear', 'Conv2d']:
                    _sample = self.sample(layer) * sqrt(temperature)
                    self._replace(_sample, layer.weight, layer.bias)

    @abstractmethod
    def scale(self, num_batches: int, make_pd: bool = False) -> None:
        """Scales the state to be the expectation over the batches.

        Abstract method to be implemented by each derived class individually.

        Args:
            num_batches: The number of batches for which the expectation should be computed.
            make_pd: Makes the result positive definite
        """
        raise NotImplementedError

    @abstractmethod
    def _quadratic_term_weight(self, weight_diff, state) -> Tensor:
        raise NotImplementedError

    def get_quadratic_term(self,
                           other_model: Module,
                           other_mean: Optional[Dict] = None,
                           scalings: List[Union[Dict, float]] = None,
                           weight_decays: Optional[List[Union[Dict, float]]] = None,
                           weight_decay_layer_names: Optional[List[str]] = None,
                           default_weight_decay: Optional[float] = None) -> Tensor:
        """Computes the quadratic term from a normal prior.

         Let :math:`\\theta` be the parameters of other_model and :math:`\\hat{\\theta}` either the model_state of self
         or other_mean if provided. Moreover, let :math:`\\Sigma` be the curvature of self, then this method computes
         .. math:
            \frac{1}{\\tau}(\\theta - \\hat{\\theta})^T \\Sigma (\\theta - \\hat{\\theta})
        :math:`\\Sigma_l = default_weight_decay I` for all layers in weight_decay_layer_names.

        Args:
            other_model: The model :math:`\\theta` as a Module.
            other_mean: The mean :math:`\\hat{\theta}` as a state dict.
            scalings: How the curvature should be scaled (each indexed by the modules of other_model).
            weight_decays: Optional L2-regularization strength (each indexed by the modules of other_model).
            weight_decay_layer_names: Layers that use an isotropic Gaussian prior with default_weight_decay
            default_weight_decay: math:`\\Sigma_l = default_weight_decay`
            temperature_scaling: temperature scaling :math:`\\tau`

        Returns:
            A tensor containing the quadratic term
        """
        if other_mean is None:
            other_mean = self.model_state
        try:
            sample_param = next(iter(other_model.parameters()))
        except StopIteration:
            sample_param = next(iter(other_mean.values()))
            dtype, device = sample_param.dtype, sample_param.device
            return torch.zeros([], dtype=dtype, device=device)

        dtype, device = sample_param.dtype, sample_param.device

        self_param_to_other_mean = {param: other_mean[name]
                                    for name, param in self.model.named_parameters()}

        quadratic_term = torch.as_tensor(0., dtype=dtype, device=device)
        name_to_module = dict(self.model.named_modules())
        for name, module_other in other_model.named_modules():
            if list(module_other.parameters()) and not list(module_other.children()):
                module = name_to_module[name]
                if weight_decay_layer_names and name in weight_decay_layer_names:
                    if default_weight_decay is None:
                        warnings.warn('default_weight_decay is not specified')
                        default_weight_decay = 1.
                    quadratic_term += default_weight_decay * \
                                      sum((param ** 2).sum() for param in module_other.parameters())
                elif module in self.state.keys():
                    state = self.state[module]
                    weight_diff = module_other.weight.to(device) - self_param_to_other_mean[module.weight].to(device)
                    weight_diff = weight_diff.view(weight_diff.size(0), -1)
                    if module.bias is not None:
                        bias_diff = module_other.bias.to(device) - self_param_to_other_mean[module.bias].to(device)
                        weight_diff = torch.cat([weight_diff, bias_diff[:, None]], dim=1)
                    scale = 1.
                    if scalings is not None:
                        for scaling in scalings:
                            if scaling:
                                scale *= scaling[module_other] if isinstance(scaling, dict) else scaling
                    # we use relu to counteract divergence when the matrix has some numerical unstable tiny eigenvalues
                    quadratic_term += scale * F.relu(self._quadratic_term_weight(weight_diff, state))
                    if weight_decays is not None:
                        weight_decay_scale = 1.
                        for weight_decay in weight_decays:
                            if weight_decay:
                                weight_decay_scale *= weight_decay[module_other] if isinstance(weight_decay, dict) \
                                    else weight_decay
                        quadratic_term += weight_decay_scale * (weight_diff ** 2).sum()
                else:
                    warnings.warn(f'Module {module} has parameters that are not taken into account.')
        return quadratic_term

    @staticmethod
    @abstractmethod
    def _kl_divergence_state(inv_state1, inv_state2, temperature_scaling: Union[float, Tensor] = 1.) -> Tensor:
        raise NotImplementedError

    def kl_divergence(self,
                      other: Curvature,
                      temperature_scaling: Union[float, Dict[Module, float]] = 1.,
                      weight_decay_layer_names: Optional[List[str]] = None,
                      default_weight_decay: Optional[float] = None):
        """ Computes the Kullback-Leibler(KL)-divergence between the normal distributions defined by self and the other
        curvature.

        The layers of other in weight_decay_layer_names are treated as isotropic Gaussians with variance
        default_weight_decay. Moreover, a temperature scaling that is used for both curvatures can be defined.

        Args:
            other: The second argument of the KL-divergence as a curvature object
            temperature_scaling: The temperature scaling which scales the covariance matrices of the Gaussians
            weight_decay_layer_names: Layers that use an isotropic Gaussian prior with default_weight_decay
            default_weight_decay: Variance of the isotropic Gaussians

        Returns:
            A tensor containing the KL-divergence
        """
        out = .5 * other.get_quadratic_term(self.model,
                                            weight_decay_layer_names=weight_decay_layer_names,
                                            default_weight_decay=default_weight_decay)
        name_to_other_module = dict(other.model.named_modules())
        for name, module1 in self.model.named_modules():
            if list(module1.parameters()) and not list(module1.children()):
                if weight_decay_layer_names and name in weight_decay_layer_names:
                    assert default_weight_decay is not None
                    other_inv_state = self._eye_state(self.inv_state[module1], 1. / sqrt(default_weight_decay))
                elif module1 in self.state.keys():
                    other_inv_state = other.inv_state[name_to_other_module[name]]
                else:
                    warnings.warn(f'Module {module1} has parameters that are not taken into account.')
                    continue
                temperature = temperature_scaling[module1] if isinstance(temperature_scaling, dict) \
                    else temperature_scaling
                out += F.relu(self._kl_divergence_state(self.inv_state[module1], other_inv_state,
                                                        temperature_scaling=temperature)).to(out)
        return out

    def add_and_scale(self,
                      other: 'Curvature',
                      scaling: List[Union[float, Tensor, Dict[Module, Union[float, Tensor]]]],
                      weight_decay_layer_names: Optional[List[str]] = None,
                      weight_decay: Optional[float, Tensor] = None):
        """Adds the other curvature to self and scales the result by the scaling factors.

        The layers in weight_decay_layer_names are treated as isotropic Gaussians
        with variance weight_decay.

        Args:
            other: The other curvature
            scaling: List of weights where each state is weighted by this scaling
            weight_decay_layer_names: Layers that use an isotropic Gaussian prior with default_weight_decay
            weight_decay: Factor that determines the weight of the identity matrix added
        """
        assert len(scaling) == 2, 'The length of the scaling list must be 2'

        if weight_decay_layer_names is None:
            weight_decay_layer_names = []

        out = type(self)(self.model,
                         layer_types=self.layer_types,
                         device=self.device)
        out.model_state = self.model_state
        out.remove_hooks_and_records()

        name_to_module_other = dict(other.model.named_modules())
        for name, module in self.model.named_modules():
            if module in self.state.keys():
                state_self = self.state[module]
                state_other = self._eye_state(state_self, weight_decay) if name in weight_decay_layer_names else \
                    other.state[name_to_module_other[name]]
                state_list = [state_self, state_other]
                scaling_list = [s[module] if isinstance(s, dict) else s
                                for s in scaling]
                out.state[module] = self._sum_state(state_list, scaling_list)

        return out



    @classmethod
    @abstractmethod
    def _sum_state(cls,
                   state_list: List[Any],
                   scaling: Optional[List[Union[float, Tensor]]],
                   weight_decay: Optional[Union[float, Tensor]] = None):
        """Sums multiple states and optionally adds a multiple of the identity matrix.

        Args:
            state_list: List of states for an explicit Curvature implementation.
            scaling: List of weights where each state is weighted by this scaling (same length as state_list)
            weight_decay: Factor that determines the weight of the identity matrix added.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def eigenvalues_of_mm(state_1: Any,
                          inv_state_2: Any) -> Tensor:
        """Computes the eigenvalues of the matrix product Fisher matrix by state_1 and the inverse Fisher given by
        inv_state2.

        Args:
            state_1: First state
            inv_state_2: Inverse second state
        """
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    # def _trace_of_mm_state(inv_state1, inv_state2):
    #     """Computes the trace of the matrix product of the Fisher matrix by inv_state1 and the inverse Fisher given by
    #     inv_state2.
    #
    #     Args:
    #         inv_state1: Inverse first state
    #         inv_state2: Inverse second state
    #     """
    #     raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _trace_of_mm_state(state1, inv_state2):
        """Computes the trace of the matrix product of the Fisher matrix by state1 and the inverse Fisher given by
        state2.

        Args:
            state1: Inverse first state
            state2: Inverse second state
        """
        raise NotImplementedError

    def trace_of_mm(self,
                    other: Curvature,
                    temperature_scaling: Union[float, Dict[Module, float]] = 1.) -> Tensor:
        """Computes the trace of the matrix product of the Fisher matrix of self and the other curvature.

        Args:
            other: The second argument of the KL-divergence as a curvature object

        Returns:
            A tensor containing the trace of the matrix product of the Fisher matrix of self and the other curvature.
        """
        out = 0.
        name_to_other_module = dict(other.model.named_modules())
        for name, module1 in self.model.named_modules():
            if list(module1.parameters()) and not list(module1.children()):
                if module1 in self.state.keys():
                    # other_inv_state = other.inv_state[name_to_other_module[name]]
                    other_inv_state = other.inv_state[name_to_other_module[name]]
                else:
                    warnings.warn(f'Module {module1} has parameters that are not taken into account.')
                    continue
                # out += self._trace_of_mm_state(self.inv_state[module1], other_inv_state)
                temperature = temperature_scaling[module1] if isinstance(temperature_scaling, dict) \
                    else temperature_scaling
                out += temperature * self._trace_of_mm_state(self.state[module1], other_inv_state)
        return out

    @abstractmethod
    def _eye_state(self, state, weight_decay):
        raise NotImplementedError

    def eye_like(self, weight_decay: float = 1.) -> 'Curvature':
        """Generates a curvature that corresponds to an isotropic Gaussian with variance weight_decay.

        Args:
            weight_decay: Variance of the Gaussian

        Returns:
            A curvature object where each state is math:`weight_decay I`
        """
        eye = self.__class__(self.model)
        eye.model_state = {k: torch.zeros_like(v) for k, v in self.model_state.items()}
        for module, state in self.state.items():
            eye.state[module] = self._eye_state(state, weight_decay)
            eye.inv_state[module] = self._eye_state(state, 1. / sqrt(weight_decay))
        return eye

    def scale_inverse(self, scaling: Union[float, Dict[Module, float]]):
        """Scales the inverse Fisher by a factor.

        Args:
            scaling: Scaling factor
        """
        for module, state in self.state.items():
            scale = scaling[module] if isinstance(scaling, dict) else scaling
            self.state[module] = self._scale_state(state, 1 / scale)
            if self.inv_state:
                self.inv_state[module] = self._scale_state(self.inv_state[module], sqrt(scale))


    @abstractmethod
    def _scale_state(self, state, scale):
        raise NotImplementedError


    def state_dict(self) -> Dict:
        """Returns a dictionary containing a whole state of the curvature.

        Returns:
            A dictionary containing a whole state of the curvature
        """
        module_to_name = {module: name for name, module in self.model.named_modules()}
        return {
            'class': self.__class__.__name__,
            'state': {module_to_name[module]: state for module, state in self.state.items()},
            'inv_state': {module_to_name[module]: state for module, state in self.inv_state.items()},
            'model_state': self.model_state,
            'layer_types': self.layer_types
        }

    def load_state_dict(self, state_dict):
        """Copies the whole state from state_dict into this curvature.

        Args:
            state_dict: A dict containing the state of the curvature
        """
        name_to_module = dict(self.model.named_modules())
        self.layer_types = state_dict['layer_types']

        if state_dict['state'] is not None:
            for name, state in state_dict['state'].items():
                module = name_to_module[name]
                assert module._get_name() in self.layer_types, f'module {module} with name {name} is not in layer_types'
                self.state[module] = state

        if state_dict['inv_state'] is not None:
            for name, state in state_dict['inv_state'].items():
                module = name_to_module[name]
                assert module._get_name() in self.layer_types, f'module {module} with name {name} is not in layer_types'
                self.inv_state[module] = state

        self.model_state = state_dict['model_state']


class Diagonal(Curvature):
    r"""The diagonal Fisher information or Generalized Gauss Newton matrix approximation.

    It is defined as :math:`F_{DIAG}=\mathrm{diag}(F)` with `F` being the Fisher defined in the `FISHER` class.
    Code inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py.

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_

    In contrast to the official curvature library, we compute the approximation not with the gradient but with the
    forward and backward pass to obtain the per-example gradients and compute the correct diagonal approximation also
    for batch sizes larger than 1.
    """

    def update(self,
               batch_size: int):
        """Computes the diagonal src for selected layer types, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            module_class = layer._get_name()
            if module_class in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    grads = self._get_grads(layer)
                    grads = (grads.view(grads.shape[0], -1) ** 2).sum(dim=0).view(layer.weight.shape[0], -1)
                    if layer in self.state:
                        self.state[layer] += grads
                    else:
                        self.state[layer] = grads

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Inverts state.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if isinstance(add, (list, tuple)) and isinstance(multiply, (list, tuple)):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            inv = torch.reciprocal(s * value + n).sqrt()

            # This is a non-in-place version of
            # inv[inv.isposinf()] = sqrt(torch.finfo().max)
            # inv[inv.isneginf()] = -sqrt(-torch.finfo().min)
            min = torch.tensor(-sqrt(-torch.finfo().min))
            max = torch.tensor(sqrt(torch.finfo().max))
            inv = torch.maximum(torch.minimum(inv, max), min)

            self.inv_state[layer] = inv

    def sample(self,
               layer: Union[Module, str]):
        """Samples from inverted state.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        return self.inv_state[layer].new(self.inv_state[layer].size()).normal_() * self.inv_state[layer]

    def scale(self, num_batches: int, make_pd: bool = False) -> None:
        """Scales the state to be the expectation over the batches.

        Args:
            num_batches: The number of batches for which the expectation should be computed.
            make_pd: Makes the result positive definite
        """
        if make_pd:
            for key, value in self.state.items():
                value[value < 1e-6] = 1e-6
                self.state[key] = value

    def _quadratic_term_weight(self, weight_diff, state) -> Tensor:
        state = state.to(weight_diff)
        return (weight_diff.view(-1) ** 2).inner(state.view(-1))

    @staticmethod
    def _kl_divergence_state(inv_state1, inv_state2, temperature_scaling: Union[float, Tensor] = 1.) -> Tensor:
        mm = (inv_state1 / inv_state2.to(inv_state1)) ** 2
        temperature_scaling = torch.as_tensor(temperature_scaling).to(inv_state1)
        return .5 * (temperature_scaling * mm.sum()
                     - (temperature_scaling.log() + 1) * inv_state1.numel()
                     - mm.log().sum())

    @classmethod
    def _sum_state(cls, state_list: List[Tensor],
                   scaling: Optional[List[Union[float, Tensor]]],
                   weight_decay: Optional[Union[float, Tensor]] = None):
        """Sums multiple states and optionally adds a multiple of the identity matrix.

        Args:
            state_list: List of states for an explicit Curvature implementation.
            scaling: List of weights where each state is weighted by this scaling (same length as state_list)
            weight_decay: Factor that determines the weight of the identity matrix added.
        """
        assert state_list, f'state_list should at least contain one element but is {state_list}'
        if scaling is not None:
            assert len(state_list) == len(
                scaling), f'state_list should contain the same number of elements as scaling, ' \
                          f'but the number of elements are: state_list {len(state_list)}, scaling {len(scaling)}.'
            state_list = [scale * state for scale, state in zip(scaling, state_list)]

        if weight_decay is not None:
            state_list.append(weight_decay * torch.ones_like(state_list[0]))
        return torch.stack(state_list, dim=0).sum(dim=0)

    @staticmethod
    def eigenvalues_of_mm(state_1: Tensor,
                          inv_state_2: Tensor) -> Tensor:
        """Computes the eigenvalues of the matrix product Fisher matrix by state_1 and the inverse Fisher given by
        inv_state2.

        Args:
            state_1: First state
            inv_state_2: Inverse second state
        """
        return (state_1 * inv_state_2 ** 2).view(-1)

    # @staticmethod
    # def _trace_of_mm_state(inv_state1, inv_state2):
    #     """Computes the trace of the matrix product of the Fisher matrix by inv_state1 and the inverse Fisher given by
    #     inv_state2.
    #
    #     Args:
    #         inv_state1: Inverse first state
    #         inv_state2: Inverse second state
    #     """
    #     return ((inv_state2 / inv_state1) ** 2).sum()

    @staticmethod
    def _trace_of_mm_state(state1, inv_state2):
        """Computes the trace of the matrix product of the Fisher matrix by state1 and the inverse Fisher given by
        state2.

        Args:
            state1: Inverse first state
            state2: Inverse second state
        """
        return (state1.to(inv_state2) * inv_state2 ** 2).sum()

    def _eye_state(self, state, weight_decay):
        device, dtype = state.device, state.dtype
        return weight_decay * torch.ones_like(state, device=device, dtype=dtype)


    def _scale_state(self, state, scale):
        return state * scale


class BlockDiagonal(Curvature):
    r"""The block-diagonal Fisher information or Generalized Gauss Newton matrix approximation.

    It can be defined as the expectation of the outer product of the gradient of the networks loss E w.r.t. its
    weights W: :math:`F=\mathbb{E}\left[\nabla_W E(W)\nabla_W E(W)^T\right]`

    Source: `A Scalable Laplace Approximation for Neural Networks <https://openreview.net/pdf?id=Skdvd2xAZ>`_
    """

    def update(self,
               batch_size: int):
        """Computes the block-diagonal (per-layer) src selected layer types, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            module_class = layer._get_name()
            if layer._get_name() in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    grads = self._get_grads(layer)
                    grads = grads.view(grads.shape[0], -1)
                    grads = grads.T @ grads

                    # Expectation
                    if layer in self.state:
                        self.state[layer] += grads
                    else:
                        self.state[layer] = grads

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Inverts state.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, float) and not isinstance(multiply, float):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            reg = torch.diag(value.new(value.shape[0]).fill_(n))
            value_reg = s * value + reg

            value_reg = ((value_reg + value_reg.T) / 2).to(torch.double)
            inv_chol = invert_and_cholesky(value_reg)
            self.inv_state[layer] = inv_chol.to(value_reg)

    def sample(self,
               layer: Module) -> Tensor:
        """Samples from inverted state.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        x = self.inv_state[layer] @ self.inv_state[layer].new(self.inv_state[layer].shape[0]).normal_()
        return x.view(layer.weight.shape[0], -1)

    def scale(self, num_batches: int, make_pd: bool = False) -> None:
        """Scales the state to be the expectation over the batches.

        Args:
            num_batches: The number of batches for which the expectation should be computed.
            make_pd: Makes the result positive definite
        """
        if make_pd:
            for key, value in self.state.items():
                self.state[key] = check_and_make_pd(value)

    def _quadratic_term_weight(self, weight_diff, state) -> Tensor:
        weight_diff_flat = weight_diff.view(-1)
        return (weight_diff_flat[None, :] @ state.to(weight_diff) @ weight_diff_flat[:, None])[0, 0]

    @staticmethod
    def _kl_divergence_state(inv_state1, inv_state2, temperature_scaling: Union[float, Tensor] = 1.) -> Tensor:
        temperature_scaling = torch.as_tensor(temperature_scaling).to(inv_state1)
        inv_state2 = inv_state2.to(inv_state1)
        half_term1 = (inv_state2.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                      inv_state1.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        mm = torch.triangular_solve(inv_state1, inv_state2, upper=False)[0]
        term2 = temperature_scaling * mm.pow(2).sum()
        return half_term1 + .5 * (term2 - (temperature_scaling.log() + 1) * inv_state1.shape[-1])

    @classmethod
    def _sum_state(cls, state_list: List[Tensor],
                   scaling: Optional[List[Union[float, Tensor]]],
                   weight_decay: Optional[Union[float, Tensor]] = None):
        """Sums multiple states and optionally adds a multiple of the identity matrix.

        Args:
            state_list: List of states for an explicit Curvature implementation.
            scaling: List of weights where each state is weighted by this scaling (same length as state_list)
            weight_decay: Factor that determines the weight of the identity matrix added.
        """
        assert state_list, f'state_list should at least contain one element but is {state_list}'
        if scaling is not None:
            assert len(state_list) == len(
                scaling), f'state_list should contain the same number of elements as scaling, ' \
                          f'but the number of elements are: state_list {len(state_list)}, scaling {len(scaling)}.'
            state_list = [scale * state for scale, state in zip(scaling, state_list)]

        if weight_decay is not None:
            state_list.append(weight_decay * torch.eye(*state_list[0].shape, device=state_list[0].device))
        return torch.stack(state_list, dim=0).sum(dim=0)

    @staticmethod
    def eigenvalues_of_mm(state_1: Tensor,
                          inv_state_2: Tensor) -> Tensor:
        """Computes the eigenvalues of the matrix product Fisher matrix by state_1 and the inverse Fisher given by
        inv_state2.

        Args:
            state_1: First state
            inv_state_2: Inverse second state
        """
        mm = state_1 @ inv_state_2 @ inv_state_2.T
        eigvals = torch.linalg.eigvals(mm)
        if not eigvals.isreal().all():
            warnings.warn(f'Matrix is not real diagonalizable, imaginary part is {eigvals.imag}')
        return eigvals.real

    # @staticmethod
    # def _trace_of_mm_state(inv_state1, inv_state2):
    #     """Computes the trace of the matrix product of the Fisher matrix by inv_state1 and the inverse Fisher given by
    #     inv_state2.
    #
    #     Args:
    #         inv_state1: Inverse first state
    #         inv_state2: Inverse second state
    #     """
    #     return torch.trace(torch.cholesky_solve(inv_state2 @ inv_state2.T, inv_state1))

    @staticmethod
    def _trace_of_mm_state(state1, inv_state2):
        """Computes the trace of the matrix product of the Fisher matrix by state1 and the inverse Fisher given by
        state2.

        Args:
            state1: Inverse first state
            state2: Inverse second state
        """
        return torch.trace(state1.to(inv_state2) @ inv_state2 @ inv_state2.T)

    def _eye_state(self, state, weight_decay):
        device, dtype = state.device, state.dtype
        return weight_decay * torch.eye(*state.shape, device=device, dtype=dtype)


    def _scale_state(self, state, scale):
        return state * scale


class KFAC(Curvature):
    r"""The Kronecker-factored Fisher information matrix approximation.

    For a single datum, the Fisher can be Kronecker-factorized into two much smaller matrices `Q` and `H`, aka
    `Kronecker factors`, s.t. :math:`F=Q\otimes H` where :math:`Q=zz^T` and :math:`H=\nabla_a^2 E(W)` with `z` being the
    output vector of the previous layer, `a` the `pre-activation` of the current layer (i.e. the output of the previous
    layer before being passed through the non-linearity) and `E(W)` the loss. For the expected Fisher,
    :math:`\mathbb{E}[Q\otimes H]\approx\mathbb{E}[Q]\otimes\mathbb{E}[H]` is assumed, which might not necessarily be
    the case.

    Code adapted from https://github.com/Thrandis/EKFAC-pytorch/kfac.py.

    Linear: `Optimizing Neural Networks with Kronecker-factored Approximate Curvature
    <https://arxiv.org/abs/1503.05671>`_

    Convolutional: `A Kronecker-factored approximate Fisher matrix for convolutional layers
    <https://arxiv.org/abs/1602.01407>`_
    """

    def update(self,
               batch_size: int):
        """Computes the 1st and 2nd Kronecker factor `Q` and `H` for each selected layer type, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            module_class = layer._get_name()
            if layer._get_name() in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    forward, backward = self.record[layer]

                    # 1st factor: Q
                    if module_class == 'Conv2d':
                        forward = F.unfold(forward, layer.kernel_size, padding=layer.padding, stride=layer.stride)
                        forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                    else:
                        forward = forward.data.t()
                    if layer.bias is not None:
                        ones = torch.ones_like(forward[:1])
                        forward = torch.cat([forward, ones], dim=0)
                    first_factor = torch.mm(forward, forward.t())

                    first_factor[first_factor.isinf()] = 1 / torch.finfo(first_factor.dtype).resolution

                    # 2nd factor: H
                    if module_class == 'Conv2d':
                        backward = backward.data.permute(1, 0, 2, 3).contiguous().view(backward.shape[1], -1)
                    else:
                        backward = backward.data.t()
                    second_factor = torch.mm(backward, backward.t()) / float(backward.shape[1])

                    # Expectation
                    if layer in self.state:
                        self.state[layer][0] += first_factor
                        self.state[layer][1] += second_factor
                    else:
                        self.state[layer] = [first_factor, second_factor]

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Inverts state.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        # assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if not isinstance(add, (float, int)) and not isinstance(multiply, (float, int)):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = float(add), float(multiply)
            first, second = value

            diag_frst = torch.diag(first.new(first.shape[0]).fill_(n ** 0.5))
            diag_scnd = torch.diag(second.new(second.shape[0]).fill_(n ** 0.5))

            reg_frst = s ** 0.5 * first + diag_frst
            reg_scnd = s ** 0.5 * second + diag_scnd

            reg_frst = ((reg_frst + reg_frst.t()) / 2.0).to(torch.double)
            reg_scnd = ((reg_scnd + reg_scnd.t()) / 2.0).to(torch.double)

            chol_ifrst = invert_and_cholesky(reg_frst)
            chol_iscnd = invert_and_cholesky(reg_scnd)

            chol_ifrst, chol_iscnd = chol_ifrst.to(first), chol_iscnd.to(second)

            self.inv_state[layer] = [chol_ifrst, chol_iscnd]

    def sample(self,
               layer: Module) -> Tensor:
        """Samples from inverted state.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        first, second = self.inv_state[layer]
        z = torch.randn(first.size(0), second.size(0), device=first.device, dtype=first.dtype)
        return (first @ z @ second.t()).t()  # Final transpose because PyTorch uses channels first

    def scale(self, num_batches: int, make_pd: bool = False) -> None:
        """Scales the state to be the expectation over the batches.

        Args:
            num_batches: The number of batches for which the expectation should be computed.
            make_pd: Makes the result positive definite
        """

        for key, value in self.state.items():
            self.state[key][0] = value[0]
            # self.state[key][0] = value[0] / num_batches # for expectation
            self.state[key][1] = value[1] / num_batches

            if make_pd:
                for i in range(2):
                    self.state[key][i] = check_and_make_pd(value[i])

    def _quadratic_term_weight(self, weight_diff, state) -> Tensor:
        state = [s.to(weight_diff) for s in state]
        return torch.tensordot(state[1] @ weight_diff @ state[0], weight_diff)

    @staticmethod
    def _kl_divergence_state(inv_state1, inv_state2, temperature_scaling: Union[float, Tensor] = 1.) -> Tensor:
        # `Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors
        # <https://arxiv.org/pdf/1603.04733.pdf>`_ equation 15
        inv_state1_left, inv_state1_right = inv_state1
        temperature_scaling = torch.as_tensor(temperature_scaling).to(inv_state1_left)
        dtype = inv_state1_left.dtype
        inv_state1_left, inv_state1_right = inv_state1_left.to(torch.double), inv_state1_right.to(torch.double)
        inv_state2_left, inv_state2_right = inv_state2[0].to(inv_state1_left), inv_state2[1].to(inv_state1_left)

        left_size, right_size = inv_state1_left.shape[-1], inv_state1_right.shape[-1]

        state2_left = torch.linalg.inv(inv_state2_left @ inv_state2_left.T)
        trace_left = torch.trace(state2_left @ inv_state1_left @ inv_state1_left.T)

        state2_right = torch.linalg.inv(inv_state2_right @ inv_state2_right.T)
        trace_right = torch.trace(state2_right @ inv_state1_right @ inv_state1_right.T)

        half_term1 = (right_size * inv_state2_left.diagonal(dim1=-2, dim2=-1).log().nansum(-1) +
                      left_size * inv_state2_right.diagonal(dim1=-2, dim2=-1).log().nansum(-1) -
                      right_size * inv_state1_left.diagonal(dim1=-2, dim2=-1).log().nansum(-1) -
                      left_size * inv_state1_right.diagonal(dim1=-2, dim2=-1).log().nansum(-1))

        trace = trace_left * trace_right
        size = left_size * right_size
        return (half_term1 + .5 * (temperature_scaling * trace - (temperature_scaling.log() + 1) * size)).to(dtype)

    @classmethod
    def _sum_state(cls, state_list: List[Tuple[Tensor, Tensor]],
                   scaling: Optional[List[Union[float, Tensor]]],
                   weight_decay: Optional[Union[float, Tensor]] = None):
        """Sums multiple states and optionally adds a multiple of the identity matrix.

        Args:
            state_list: List of states for an explicit Curvature implementation.
            scaling: List of weights where each state is weighted by this scaling (same length as state_list)
            weight_decay: Factor that determines the weight of the identity matrix added.
        """
        assert state_list, f'state_list should at least contain one element but is {state_list}'

        shape_left, dtype, device = state_list[0][0].shape, state_list[0][0].dtype, state_list[0][0].device
        shape_right = state_list[0][1].shape

        state_list_left = [state[0].to(device) for state in state_list]
        state_list_right = [state[1].to(device) for state in state_list]

        if scaling is not None:
            assert len(state_list) == len(
                scaling), f'state_list should contain the same number of elements as scaling, ' \
                          f'but the number of elements are: state_list {len(state_list)}, scaling {len(scaling)}.'
            eps = torch.finfo(torch.double).eps
            state_list_left = [torch.sqrt(torch.as_tensor(scale).to(state) + eps) * state for scale, state in
                               zip(scaling, state_list_left)]
            state_list_right = [torch.sqrt(torch.as_tensor(scale).to(state) + eps) * state for scale, state in
                                zip(scaling, state_list_right)]

        if weight_decay is not None and weight_decay != 0.:
            state_list_left.append(
                torch.sqrt(torch.as_tensor(weight_decay, dtype=dtype, device=device)) * torch.eye(*shape_left,
                                                                                                  dtype=dtype,
                                                                                                  device=device))
            state_list_right.append(
                torch.sqrt(torch.as_tensor(weight_decay, dtype=dtype, device=device)) * torch.eye(*shape_right,
                                                                                                  dtype=dtype,
                                                                                                  device=device))
        left_tensor = torch.stack(state_list_left)
        right_tensor = torch.stack(state_list_right)

        error_in_sum = False
        try:
            out = sum_kronecker_products(left_tensor, right_tensor, assert_positive_definite=False)
        except RuntimeError:
            error_in_sum = True
        if error_in_sum or out[0].isnan().any() or out[1].isnan().any():
            out = power_method_sum_kronecker_products_full_rank(left_tensor, right_tensor,
                                                                assert_positive_definite=False)
            # out = power_method_sum_kronecker_products_full_rank(left_tensor, right_tensor, assert_positive_definite=True)
        return list(out)

    @staticmethod
    def eigenvalues_of_mm(state_1: Union[List[Tensor], Tuple[Tensor]],
                          inv_state_2: Union[List[Tensor], Tuple[Tensor]]) -> Tensor:
        """Computes the eigenvalues of the matrix product Fisher matrix by state_1 and the inverse Fisher given by
        inv_state2.

        Args:
            state_1: First state
            inv_state_2: Inverse second state
        """
        left_1, right_1 = state_1
        dtype = left_1.dtype
        left_1, right_1 = left_1.to(torch.double), right_1.to(torch.double)
        inv_left_2, inv_right_2 = inv_state_2
        inv_left_2, inv_right_2 = inv_left_2.to(torch.double), inv_right_2.to(torch.double)
        inv_left_2, inv_right_2 = inv_left_2.to(left_1.device), inv_right_2.to(right_1.device)

        eps = torch.finfo().eps

        left_1 = check_and_make_pd(left_1)
        inv_left_2 = inv_left_2 @ inv_left_2.T
        inv_left_2 = check_and_make_pd(inv_left_2)
        left_mm = left_1 @ inv_left_2
        left_eigvals = torch.linalg.eigvals(left_mm)
        if not (left_eigvals.imag < eps).all():
            warnings.warn(f'Left matrix is not real diagonalizable, imaginary part is {left_eigvals.imag}')
        left_eigvals = left_eigvals.real

        right_1 = check_and_make_pd(right_1)
        inv_right_2 = inv_right_2 @ inv_right_2.T
        inv_right_2 = check_and_make_pd(inv_right_2)
        right_mm = right_1 @ inv_right_2
        right_eigvals = torch.linalg.eigvals(right_mm)
        if not (right_eigvals.imag < eps).all():
            warnings.warn(f'Right matrix is not real diagonalizable, imaginary part is {right_eigvals.imag}')
        right_eigvals = right_eigvals.real

        return left_eigvals.outer(right_eigvals).view(-1).contiguous().to(dtype)

    # @staticmethod
    # def _trace_of_mm_state(inv_state1, inv_state2):
    #     """Computes the trace of the matrix product of the Fisher matrix by inv_state1 and the inverse Fisher given by
    #     inv_state2.
    #
    #     Args:
    #         inv_state1: Inverse first state
    #         inv_state2: Inverse second state
    #     """
    #     inv_left1, inv_right1 = inv_state1
    #     inv_left2, inv_right2 = inv_state2
    #
    #     out_left = torch.trace(torch.cholesky_solve(inv_left2 @ inv_left2.T, inv_left1))
    #     out_right = torch.trace(torch.cholesky_solve(inv_right2 @ inv_right2.T, inv_right1))
    #     return out_left * out_right

    # @staticmethod
    # def _trace_of_mm_state(state1, state2):
    #     """Computes the trace of the matrix product of the Fisher matrix by state1 and the inverse Fisher given by
    #     state2.
    #
    #     Args:
    #         state1: Inverse first state
    #         state2: Inverse second state
    #     """
    #     left1, right1 = state1
    #     left2, right2 = state2
    #
    #     # if not positive_definite.check(left2):
    #     #     min_eigval = torch.linalg.eigvalsh(left2).min()
    #     #     left2 = left2 + \
    #     #                 torch.eye(left2.shape[0], device=left2.device) * \
    #     #                 (torch.finfo(torch.float).eps - 2 * min_eigval)
    #     # reg_left2 = left2 + torch.eye(left2.shape[0], device=left2.device) * torch.finfo(torch.float).eps
    #     left2 = check_and_make_pd(left2)
    #     # out_left = torch.trace(left1 @ torch.linalg.pinv(left2))
    #     out_left = torch.trace(torch.linalg.solve(left2.T, left1.T).T)
    #
    #     # reg_right2 = right2 + torch.eye(right2.shape[0], device=right2.device) * torch.finfo(torch.float).eps
    #     right2 = check_and_make_pd(right2)
    #     # out_right = torch.trace(right1 @ torch.linalg.pinv(right2))
    #     # if not positive_definite.check(right2):
    #     #     min_eigval = torch.linalg.eigvalsh(right2).min()
    #     #     right2 = right2 + \
    #     #                 torch.eye(right2.shape[0], device=right2.device) * \
    #     #                 (torch.finfo(torch.float).eps - 2 * min_eigval)
    #     out_right = torch.trace(torch.linalg.solve(right2.T, right1.T).T)
    #     return out_left * out_right

    @staticmethod
    def _trace_of_mm_state(state1, inv_state2):
        """Computes the trace of the matrix product of the Fisher matrix by state1 and the inverse Fisher given by
        inv_state2.

        Args:
            state1: Inverse first state
            inv_state2: Inverse second state
        """
        left1, right1 = state1
        inv_chol_left2, inv_chol_right2 = inv_state2

        out_left = torch.trace(left1.to(inv_chol_left2) @ inv_chol_left2 @ inv_chol_left2.T)
        out_right = torch.trace(right1.to(inv_chol_right2) @ inv_chol_right2 @ inv_chol_right2.T)
        return out_left * out_right

    def _eye_state(self, state, weight_decay):
        device, dtype = state[0].device, state[0].dtype
        return [torch.eye(*state[0].shape, device=device, dtype=dtype),
                weight_decay * torch.eye(*state[1].shape, device=device, dtype=dtype)]


    def _scale_state(self, state, scale):
        return [sqrt(scale) * state[0], sqrt(scale) * state[1]]


class KFOC(KFAC):
    r"""The Kronecker-factored Fisher information matrix optimal approximation.

    For a single datum, the Fisher can be Kronecker-factorized into two much smaller matrices `L` and `R`, aka
    `Kronecker factors`, s.t. :math:`F=L\otimes R` where :math:`L=zz^T` and :math:`R=\nabla_a^2 E(W)` with `z` being the
    output vector of the previous layer, `a` the `pre-activation` of the current layer (i.e. the output of the previous
    layer before being passed through the non-linearity) and `E(W)` the loss. For the expectation of the Fisher, the
    optimal Kronecker factorization is computed for a whole batch using the power method.

    Official implementation of `Kronecker-Factored Optimal Curvature
    <http://bayesiandeeplearning.org/2021/papers/33.pdf>`_
    """

    def __init__(self,
                 model: Union[Module, Sequential],
                 layer_types: Union[List[str], str] = None,
                 device: Optional[torch.device] = None,
                 approx: bool = False):
        """K-FOC class initializer.

        For the recursive computation of `H`, outputs and inputs for each layer are recorded in `record`. Forward and
        backward hook handles are stored in `hooks` for subsequent removal.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            layer_types: Types of layers for which to compute src information. Supported are `Linear`, `Conv2d`.
                If `None`, all supported types are considered. Default: None.
            device: device on which the curvature should be computed
            approx: True iff the K-FAC approximation should be used to compute else the running average is approximated
        """
        super().__init__(model, layer_types, device)
        self.approx = approx

    def update(self,
               batch_size: int):
        """Computes the 1st and 2nd Kronecker factor `L` and `R` for each selected layer type, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            module_class = layer._get_name()
            if layer._get_name() in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    forward, backward = self.record[layer]

                    # 1st factor: Q
                    forward = self._transform_forward(layer, forward)

                    # 2nd factor: H
                    if module_class == 'Conv2d':
                        backward = backward.data.view(*backward.shape[:-2], -1).permute(0, 2, 1)
                    else:
                        backward = backward.data

                    if module_class == 'Conv2d':
                        first_factor, second_factor = power_method_sum_kronecker_products_rank_1(forward, backward)
                    else:
                        first_factor, second_factor = power_method_sum_kronecker_products_rank_1(forward[:, None, :],
                                                                                                 backward[:, None, :])

                    # Expectation
                    if layer in self.state:
                        if self.approx:
                            self.state[layer][0] += first_factor
                            self.state[layer][1] += second_factor
                        else:
                            left_tensor = torch.stack([self.state[layer][0], first_factor])
                            right_tensor = torch.stack([self.state[layer][1], second_factor])
                            try:
                                self.state[layer] = list(
                                    sum_kronecker_products(left_tensor,
                                                           right_tensor,
                                                           assert_positive_definite=False))
                            except RuntimeError:
                                self.state[layer] = list(
                                    power_method_sum_kronecker_products_full_rank(left_tensor,
                                                                                  right_tensor,
                                                                                  assert_positive_definite=False))
                    else:
                        self.state[layer] = [first_factor, second_factor]

    def scale(self, num_batches: int, make_pd: bool = False) -> None:
        """Scales the state to be the expectation over the batches.

        Args:
            num_batches: The number of batches for which the expectation should be computed.
            make_pd: Makes the result positive definite
        """

        for key, value in self.state.items():
            if self.approx:
                self.state[key][1] = value[1] / num_batches

            if make_pd:
                for i in range(2):
                    self.state[key][i] = check_and_make_pd(value[i])