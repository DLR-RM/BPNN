"""Various curvature scalings for Bayesian Progressive Neural Networks"""
from abc import ABC, abstractmethod
from math import log
from typing import List, Union, Any, Optional, Dict, Tuple, TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .utils import optimize, run_epoch, mc_allester_bound, catoni_bound

if TYPE_CHECKING:
    from .bpnn import BayesianProgressiveNeuralNetwork


class CurvatureScaling(ABC):
    """Base class for all curvature scalings.

    Curvature scalings compute the alpha and beta values for all or individual
    layers. These values determine how the curvature of the likelihood and prior
    are combined.
    """

    def reset(self, *args: Any, **kwargs: Any):
        """Resets the main parameters.

        No parameters are updated here.
        """
        pass

    @abstractmethod
    def find_optimal_scaling(self, *args: Any, **kwargs: Any) \
            -> Tuple[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]]]:
        """Computes the scales.

        Returns:
            alpha: The prior scale
            beta: The likelihood scale
        """
        raise NotImplementedError


class ScalarScaling(CurvatureScaling):
    """Fixed scaling given as input."""

    def __init__(
            self,
            alpha: Union[Dict[Module, float], float],
            beta: Union[Dict[Module, float], float],
            temperature_scaling: Union[Dict[Module, float], float],
            *args: Any, **kwargs: Any):
        """ScalarScaling initializer.

        Args:
            alpha: The prior scale
            beta: The likelihood scales
            temperature_scaling: The temperature scaling
        """
        self.alpha = alpha
        self.beta = beta
        self.temperature_scaling = temperature_scaling

    def find_optimal_scaling(self, *args: Any, **kwargs: Any) \
            -> Tuple[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]]]:
        """Computes the scales.

        Returns:
            alpha: The prior scale
            beta: The likelihood scale
            temperature_scaling: The temperature scaling
        """
        return self.alpha, self.beta, self.temperature_scaling


class StandardBayes(ScalarScaling):
    """Scales prior and likelihood with one.

    This is equivalent to the standard Bayes scaling
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """StandardBayes initializer."""
        super().__init__(1., 1., 1., *args, **kwargs)


class ValidationScaling(CurvatureScaling):
    """Scales prior and likelihood with validation metric."""

    def __init__(
            self,
            alphas: List[float],
            betas: List[float],
            temperature_scalings: List[float],
            num_samples: int = 100,
            dataloader: torch.utils.data.DataLoader = None,
            criterion: torch.nn.Module = None,
            is_classification: bool = True,
            metric: str = 'accuracy',
            metric_should_be_large: bool = True,
            verbose: bool = False,
            *args: Any,
            **kwargs: Any):
        """ValidationScaling initializer.

        Args:
            alphas: The prior scales that should be tried
            betas: The likelihood scales that should be tried
            temperature_scalings: The temperature scales that should be tried
            num_samples: The number of samples to use for the validation metric
            dataloader: The data loader
            criterion: The criterion to use
            is_classification: Whether the task is classification or regression
            metric: The metric to use for scaling
            metric_should_be_large: Whether the metric should be large or small
            verbose: Whether to print the best metric
        """
        super().__init__()
        self.alphas = alphas
        self.betas = betas
        self.temperature_scalings = temperature_scalings  #
        self.num_samples = num_samples
        self.dataloader = dataloader
        self.criterion = criterion
        self.is_classification = is_classification
        self.metric = metric
        self.metric_should_be_large = metric_should_be_large
        self.verbose = verbose

    def reset(
            self,
            alphas: Optional[List[float]] = None,
            betas: Optional[List[float]] = None,
            temperature_scalings: Optional[List[float]] = None,
            dataloader: Optional[torch.utils.data.DataLoader] = None,
            criterion: Optional[torch.nn.Module] = None,
            is_classification: Optional[bool] = None,
            metric: Optional[str] = None,
            metric_should_be_large: Optional[bool] = None,
            verbose: Optional[bool] = None,
            *args: Any,
            **kwargs: Any):
        """Resets the main parameters.

        Args:
            alphas: The prior scales that should be tried
            betas: The likelihood scales that should be tried
            temperature_scalings: The temperature scales that should be tried
            dataloader: The validation dataloader
            criterion: The criterion to use
            is_classification: Whether the task is classification or regression
            metric: The metric to use for the scaling
            metric_should_be_large: Whether the metric should be large or small
            verbose: Whether to print the best metric
        """
        if alphas is not None:
            self.alphas = alphas
        if betas is not None:
            self.betas = betas
        if temperature_scalings is not None:
            self.temperature_scalings = temperature_scalings
        if dataloader is not None:
            self.dataloader = dataloader
        if criterion is not None:
            self.criterion = criterion
        if is_classification is not None:
            self.is_classification = is_classification
        if metric is not None:
            self.metric = metric
        if metric_should_be_large is not None:
            self.metric_should_be_large = metric_should_be_large
        if verbose is not None:
            self.verbose = verbose

    @torch.no_grad()
    def find_optimal_scaling(
            self,
            update: Callable[[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
                              Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
                              Union[float, Tensor, Dict[Module, Union[float, Tensor]]]], None],
            model: torch.nn.Module,
            *args: Any,
            **kwargs: Any) \
            -> Tuple[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]]]:
        """Computes the scales.

        Args:
            update: The update function
            model: The model to optimize

        Returns:
            alpha: The prior scale
            beta: The likelihood scale
            temperature_scaling: The temperature scaling
        """
        best_alpha = np.NaN
        best_beta = np.NaN
        best_temperature_scaling = np.NaN
        best_metric = -np.inf if self.metric_should_be_large else np.inf
        is_training = model.training
        model.eval()
        if hasattr(model, 'eval_num_samples'):
            eval_num_samples = model.eval_num_samples
            model.eval_num_samples = self.num_samples
        for alpha in self.alphas:
            for beta in self.betas:
                for temperature_scaling in self.temperature_scalings:
                    update(alpha, beta, temperature_scaling)
                    metrics = run_epoch(
                        model, self.dataloader, self.criterion,
                        is_classification=self.is_classification,
                        optimizer=None, train=False,
                        metrics=[self.metric], return_if_loss_nan_or_inf=True)
                    if np.isnan(metrics['loss']) or np.isinf(metrics['loss']):
                        continue
                    metric = metrics[self.metric]
                    if (self.metric_should_be_large and metric > best_metric) or \
                            (not self.metric_should_be_large and metric < best_metric):
                        best_alpha = alpha
                        best_beta = beta
                        best_temperature_scaling = temperature_scaling
                        best_metric = metric
                        if self.verbose:
                            print(
                                f'New best: alpha: {best_alpha:.2f}, beta: '
                                f'{best_beta:.2f}, {best_temperature_scaling:.2f},'
                                f' metric: {best_metric:.2f}')
        if hasattr(model, 'eval_num_samples'):
            model.eval_num_samples = eval_num_samples
        model.train(is_training)
        return best_alpha, best_beta, best_temperature_scaling


class PACBayesScaling(CurvatureScaling):
    """Base class for curvature scalings that use PAC-Bayesian bounds.

    The PAC-Bayesian bounds are optimized to find the scales that minimize the
    generalization bounds.
    """

    def __init__(
            self,
            confidence: float = .8,
            fixed: Optional[Tuple[Optional[Union[Dict[Module, float], float]],
            Optional[Union[Dict[Module, float], float]],
            Optional[Union[Dict[Module, float], float]]]] = None,
            shared: Tuple[bool, bool, bool] = False,
            **kwargs: Any):
        """PACBayesScaling initializer.

        Args:
            confidence: The bound of the probability in the PAC-Bayesian bounds
            fixed: The fixed scales
            shared: Whether the scales should be shared
        """
        super().__init__()
        self.confidence = confidence
        if fixed is None:
            fixed = (None, None, None)

        assert any(f is None for f in fixed), \
            'Cannot have both fixed_alpha and fixed_beta. Use instead ScalarScaling.'

        self.fixed_alpha, self.fixed_beta, self.fixed_temperature = fixed

        self.shared_alpha, self.shared_beta, self.shared_temperature = shared

    @staticmethod
    def _compute_expected_empirical_risk(
            trace: Callable[
                [Union[float, Tensor, Dict[Module, Union[float, Tensor]]]], Tensor],
            temperature_scaling: Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            len_data: int,
            negative_data_log_likelihood: Union[float, Tensor],
            scaled_fixed_model_size: float) -> Tensor:
        """Computes the upper bound of the expected empirical risk of the model.

        Args:
            trace: The trace of the model
            temperature_scaling: The temperature scaling
            len_data: The length of the train dataset
            negative_data_log_likelihood: The negative data log likelihood on the train dataset
            scaled_fixed_model_size: How many parameters were previously fixed
        """
        out = (negative_data_log_likelihood +
               (scaled_fixed_model_size + trace(temperature_scaling)) / 2) \
              / (len_data * log(2))
        return out

    def _objective(
            self,
            kl_divergence: Tensor,
            expected_empirical_risk: float,
            additional_params: List[Tensor],
            len_data: int) -> Tensor:
        """Computes the objective function.

        Args:
            kl_divergence: The KL divergence of the model
            expected_empirical_risk: The empirical risk of the model
            additional_params: Additional parameters to use in the objective
            len_data: The length of the train dataset
        """
        raise NotImplementedError

    def _get_additional_params(
            self,
            device: torch.device,
            dtype: torch.dtype) -> List[Tensor]:
        """Returns the additional parameters to use in the objective function."""
        raise NotImplementedError

    def find_optimal_scaling(
            self,
            update: Callable[[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
                              Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
                              Union[float, Tensor, Dict[Module, Union[float, Tensor]]]], None],
            trace: Callable[[Union[float, Tensor, Dict[Module, Union[float, Tensor]]]], Tensor],
            kl_divergence: Callable[[], Tensor],
            modules: List[Module],
            len_data: int,
            negative_data_log_likelihood: Union[float, Tensor],
            *args: Any,
            scaled_fixed_model_size: Optional[float] = 0,
            max_iter: int = 500,
            eps: float = 1e-6,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs: Any) \
            -> Tuple[Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]],
            Union[float, Tensor, Dict[Module, Union[float, Tensor]]]]:
        """Computes the scales.

        The layers of other in weight_decay_layer_names are treated as isotropic
        Gaussians with variance weight_decay. Moreover, a temperature scaling
        that is used for both curvatures can be defined.

        Args:
            update: A function that updates the parameters of the model
            trace: A function that computes the trace of the curvature
            kl_divergence: A function that computes the KL divergence of the model
            modules: The modules to compute the scales for
            len_data: The length of the train dataset
            negative_data_log_likelihood: The negative data log likelihood on the
                train dataset
            scaled_fixed_model_size: How many parameters were previously fixed
            max_iter: The maximal number of iterations
            eps: The optimization is stopped if all gradients are smaller than
                this value
            device: The device to use
            dtype: The dtype to use

        Returns:
            alpha: The prior scale
            beta: The likelihood scale
            temperature_scaling: The temperature scaling
        """

        def apply(param: Tensor, to_float: bool = False, fn: Callable = torch.exp) -> Dict[Module, Tensor]:
            if param.ndim == 0:
                return fn(param).item() if to_float else fn(param)
            else:
                return {module: fn(p).item() if to_float else fn(p)
                        for module, p in zip(modules, param)}

        def objective(log_alpha, log_beta, log_temperature_scaling, *additional_params):
            temperature_scaling = apply(log_temperature_scaling, fn=torch.exp)
            update(apply(log_alpha), apply(log_beta), temperature_scaling)
            kl = kl_divergence(temperature_scaling)
            expected_empirical_risk = self._compute_expected_empirical_risk(
                trace, temperature_scaling,
                len_data, negative_data_log_likelihood,
                scaled_fixed_model_size)
            return self._objective(kl, expected_empirical_risk, additional_params, len_data=len_data)

        def init_and_add_param(
                flexible_params: List[Tensor],
                param: Optional[Union[Dict[Module, float], float]] = None,
                share_param: bool = False) -> Tensor:
            if param is not None:
                if isinstance(param, dict):
                    out = torch.tensor(
                        [param[module] for module in modules],
                        dtype=dtype, device=device).log()
                else:
                    out = torch.tensor(param, dtype=dtype, device=device).log()
            else:
                if share_param:
                    out = torch.zeros([], dtype=dtype, device=device)
                else:
                    out = torch.zeros(len(modules), dtype=dtype, device=device)
                flexible_params.append(out)
            return out

        flexible_params = []
        log_alpha = init_and_add_param(flexible_params, self.fixed_alpha, self.shared_alpha)
        log_beta = init_and_add_param(flexible_params, self.fixed_beta, self.shared_beta)
        log_temperature_scaling = init_and_add_param(flexible_params, self.fixed_temperature, self.shared_temperature)
        additional_params = self._get_additional_params(device, dtype)
        flexible_params.extend(additional_params)

        params = [log_alpha, log_beta, log_temperature_scaling] + additional_params

        params = optimize(params, flexible_params, objective, max_iter=max_iter, eps=eps)
        log_alpha, log_beta, log_temperature_scaling = params[:3]

        return apply(log_alpha, True), apply(log_beta, True), apply(log_temperature_scaling, True, torch.exp)


class McAllesterScaling(PACBayesScaling):

    def _objective(
            self,
            kl_divergence: Tensor,
            expected_empirical_risk: Tensor,
            additional_params: List[Tensor],
            len_data: int) -> Tensor:
        """Computes the objective function.

        Args:
            kl_divergence: The KL divergence of the model
            expected_empirical_risk: The expected empirical risk of the model
            additional_params: Additional parameters to use in the objective
            len_data: The length of the train dataset
        """
        return mc_allester_bound(
            expected_empirical_risk, kl_divergence,
            len_data=len_data, confidence=self.confidence)

    def _get_additional_params(
            self,
            device: torch.device,
            dtype: torch.dtype) -> List[Tensor]:
        """Returns the additional parameters to use in the objective function."""
        return []


class CatoniScaling(PACBayesScaling):

    def _objective(
            self,
            kl_divergence: Tensor,
            expected_empirical_risk: Tensor,
            additional_params: List[Tensor],
            len_data: int) -> Tensor:
        """Computes the objective function.

        Args:
            kl_divergence: The KL divergence of the model
            expected_empirical_risk: The expected empirical risk of the model
            additional_params: Additional parameters to use in the objective
            len_data: The length of the train dataset
        """
        return catoni_bound(
            expected_empirical_risk, kl_divergence,
            len_data=len_data, log_catoni_scale=additional_params[0],
            confidence=self.confidence, return_log_catoni_scale=False)

    def _get_additional_params(
            self,
            device: torch.device,
            dtype: torch.dtype) -> List[Tensor]:
        """Returns the additional parameters to use in the objective function."""
        log_catoni_scale = torch.ones([], dtype=dtype, device=device).log()
        return [log_catoni_scale]
