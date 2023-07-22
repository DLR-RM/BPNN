"""Various objective functions for Bayesian Progressive Neural Networks"""
from abc import ABC, abstractmethod
from math import log
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn

from .utils import mc_allester_bound, catoni_bound

if TYPE_CHECKING:
    from .bpnn import BayesianProgressiveNeuralNetwork


class Criterion(nn.Module, ABC):
    """Base class for all criterions.

    All criterions have a corresponding model, a cross-entropy loss and the length of the data.
    """

    def __init__(self, model: nn.Module,
                 loss_function: nn.Module = nn.CrossEntropyLoss(),
                 **kwargs):
        """Criterion initializer.

        Args:
            model: A PyTorch model
        """
        super().__init__()
        assert hasattr(model, 'get_quadratic_term'), 'model should implement a function get_quadratic_term'
        self.model = model
        self.loss_function = loss_function
        self.register_buffer('len_data', None)

    @abstractmethod
    def forward(self, logits, targets):
        raise NotImplementedError


class ScaledMaximumAPosterioriCriterion(Criterion):
    """Maximum a posteriori objective, where the prior term is scaled."""

    def __init__(self,
                 model: nn.Module,
                 scale: float,
                 loss_function: nn.Module = nn.CrossEntropyLoss(),
                 **kwargs):
        """ScaledMaximumAPosterioriCriterion initializer.

        Args:
            model: A PyTorch model
            scale: The scaling of the prior term
        """
        super().__init__(model, loss_function, **kwargs)
        self.scale = scale

    def forward(self, logits, targets):
        nll = self.loss_function(logits, targets)
        prior = .5 * logits.shape[0] * self.model.get_quadratic_term() / self.len_data
        return nll + self.scale * prior


class MaximumAPosterioriCriterion(ScaledMaximumAPosterioriCriterion):
    """Maximum a posteriori objective.

    Is equivalent to ScaledMaximumAPosterioriCriterion with scale = 1.
    """

    def __init__(self,
                 model: nn.Module,
                 loss_function: nn.Module = nn.CrossEntropyLoss(),
                 **kwargs):
        """MaximumAPosterioriCriterion initializer.

        Args:
            model: A PyTorch model
        """
        super().__init__(model, scale=1., loss_function=loss_function, **kwargs)


class PACBayesCriterion(Criterion, ABC):
    """Base class for criterions that use PAC-Bayesian bounds.

    Because the curvature dependent terms are not known, only the quadratic term
    is used and all other terms are assumed to be zero.
    """

    def __init__(self,
                 model: 'BayesianProgressiveNeuralNetwork',
                 confidence: float = .95,
                 **kwargs):
        """PACBayesCriterion initializer.

        Args:
            model: A PyTorch model
            confidence: The bound of the probability in the PAC-Bayesian bounds
        """
        super().__init__(model, nn.CrossEntropyLoss(), **kwargs)
        self.register_buffer('confidence', torch.as_tensor(confidence))


class McAllesterCriterion(PACBayesCriterion):
    """Uses the McAllester bound."""

    def forward(self, logits, targets):
        nll = self.loss_function(logits, targets)
        trace_fixed = self.model.fixed_model_size_trace
        num_parameters_current = sum(p.numel() for p in self.model.networks[-1].parameters())
        expected_empirical_risk = (nll + (trace_fixed + num_parameters_current) / (2 * self.len_data)) / log(2)
        quadratic_term = .5 * self.model.get_quadratic_term()
        return mc_allester_bound(expected_empirical_risk, quadratic_term, self.len_data, self.confidence)


class CatoniCriterion(PACBayesCriterion):
    """Uses the Catoni bound."""

    def __init__(self,
                 model: nn.Module,
                 confidence: float = .95,
                 update_catoni_scale_every_n_inputs: Optional[int] = None,
                 verbose_updates: bool = False,
                 **kwargs):
        """MaximumAPosterioriCriterion initializer.

        Args:
            model: A PyTorch model
            confidence: The bound of the probability in the PAC-Bayesian bounds
            update_catoni_scale_every_n_inputs: Determines how often the catoni
                scale is optimized
            verbose_updates: Whether the updates should be printed
        """
        super().__init__(model, confidence, **kwargs)
        self.register_buffer('log_catoni_scale', torch.as_tensor(1.))
        self.counter = 0
        self.update_catoni_scale_every_n_inputs = update_catoni_scale_every_n_inputs
        self.verbose_updates = verbose_updates

    def forward(self, logits, targets):
        nll = self.loss_function(logits, targets)
        trace_fixed = self.model.fixed_model_size_trace
        num_parameters_current = sum(p.numel() for p in self.model.networks[-1].parameters())
        expected_empirical_risk = (nll + (trace_fixed + num_parameters_current) / (2 * self.len_data)) / log(2)
        quadratic_term = .5 * self.model.get_quadratic_term()
        reg = (quadratic_term - log(1 - self.confidence)) / self.len_data
        out = self.log_catoni_scale.exp() * expected_empirical_risk + reg
        if self.model.training:
            self.counter += logits.shape[0]
            update_catoni_scale_every_n_inputs = self.update_catoni_scale_every_n_inputs \
                if self.update_catoni_scale_every_n_inputs is not None else self.len_data.item()
            if self.counter >= update_catoni_scale_every_n_inputs:
                _, self.log_catoni_scale = catoni_bound(nll, quadratic_term, self.len_data, None, self.confidence,
                                                        return_log_catoni_scale=True, max_iter=1_000)
                if self.verbose_updates:
                    print(f'catoni_scale updated to {self.log_catoni_scale.exp()}')
                self.counter = 0
        return out
