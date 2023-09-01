"""Provides utility functions."""
import os
import sys
from abc import ABC, abstractmethod
from math import sqrt, log, pi
from typing import List, Dict, Union, Optional, Callable, Tuple, Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.curvature.curvatures import Curvature
from src.curvature.utils import seed_all_rng

seed = seed_all_rng()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

torch_data_path = os.path.join(base_path, 'data', 'torch')


def set_seed(local_seed=None):
    """Sets the global random seed.

    If ``local_seed`` is None, a random seed is chosen.

    Args:
        local_seed: The seed value to use
    """
    global seed
    seed = seed_all_rng(local_seed)
    print(f'random seed: {seed}')


def accuracy(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the accuracy.

    This method computes the argmax over the dimension 1 of the
    input and compares this prediction with the target.
    In addition, the result can be reduced with the reduction_fn

    Args:
        input: (shape [B, C]) The class input
        target: (shape [B]) The target values
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task (else outputs
            a zero tensor)

    Returns:
        A tensor containing the reduced accuracy value
    """
    if not classification:
        return torch.zeros([]).to(input)
    if reduction_fn is None:
        reduction_fn = torch.mean
    return 100.0 * reduction_fn(torch.argmax(input, dim=1) == target)


def nll_cls(input: Tensor, target: Tensor) -> Tensor:
    """Computes the negative log-likelihood for classification.

    Args:
        input: (shape [B, C]) The class probabilities
        target: (shape [B]) The target values

    Returns:
        A tensor containing the reduced negative log-likelihood value
    """
    tiny = torch.finfo(input.dtype).tiny
    correct_input = input[torch.arange(input.shape[0]), target]
    return -torch.log(correct_input + tiny)


def nll_reg(input: Tensor, target: Tensor) -> Tensor:
    """Computes the negative log-likelihood for regression.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B, C]) The target values

    Returns:
        A tensor containing the negative log-likelihood
    """
    se = torch.sum((input - target) ** 2, dim=-1)
    return 0.5 * (log(2 * pi) + se)


def negative_log_likelihood(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the negative log-likelihood.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B] classification or shape [B, C] regression)
            The target values
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: If True, the negative log-likelihood is computed for
            classification, otherwise for regression

    Returns:
        A tensor containing the reduced negative log-likelihood value
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    nll = nll_cls if classification else nll_reg
    return reduction_fn(nll(input, target))


def squared_error(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the squared error.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B, C]) The target values
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task

    Returns:
        A tensor containing the squared error
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    if classification:
        target = torch.eye(input.shape[1], device=input.device, dtype=input.dtype)[target]
    return reduction_fn(torch.mean((input - target) ** 2, dim=-1))


def absolute_error(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the absolute error.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B, C]) The target values
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task

    Returns:
        A tensor containing the absolute error
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    if classification:
        target = torch.eye(input.shape[1], device=input.device, dtype=input.dtype)[target]
    return reduction_fn(torch.mean(torch.abs(input - target), dim=-1))


def brier_score(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the Brier score.

    Args:
        input: (shape [B, C]) The class probabilities
        target: (shape [B]) The target values
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task

    Returns:
        A tensor containing the reduced brier score value
    """
    if not classification:
        return torch.tensor(torch.nan).to(input)
    if reduction_fn is None:
        reduction_fn = torch.mean
    one_hot_labels = torch.eye(input.shape[1], device=input.device, dtype=input.dtype)[target]
    return reduction_fn(torch.sum((input - one_hot_labels) ** 2, dim=1))


def entropy(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = None) -> Tensor:
    """Computes the entropy.

    Args:
        input: (shape [B, C]) The class probabilities
        target: The target values (not used)
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task

    Returns:
        The reduced entropy value
    """
    if classification is None:
        classification = True
    if reduction_fn is None:
        reduction_fn = torch.mean
    input_normalized = input / input.sum(dim=-1, keepdim=True)
    if classification:
        tiny = torch.finfo(input_normalized[0].dtype).tiny
        entropy_values = -torch.sum(input_normalized * torch.log(input_normalized + tiny), dim=-1)
    else:
        entropy_values = input[-1].shape[1] * 0.5 * (log(2 * pi) + 1) * torch.ones(input[-1].shape[0])
    return reduction_fn(entropy_values, dim=-1)


def standard_deviation(
        input: Tensor,
        target: Tensor,
        reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
        classification: bool = True) -> Tensor:
    """Computes the standard deviation.

    Args:
        input: (shape [B, N, C]) The input
        target: The target values (not used)
        reduction_fn: A reduction function that can be applied to shape [B]
        classification: Whether the task is a classification task

    Returns:
        A tensor containing the standard deviation
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    return reduction_fn(torch.std(input, dim=1))


class Metric(ABC):
    """Base class for all Metrics.

    The metric should implement ``update`` and ``summarize``.
    This allows to compute the metrics in an online setting.
    """

    assert_mean = True

    @abstractmethod
    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of class probabilities and
        target.

        Args:
            input: The class probabilities as a list of (shape [B, C]) tensors
            target: (shape [B]) The target values
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        raise NotImplementedError


class FunctionalMetric(Metric):
    """A functional metric.

    This metric applies a given function on an index of the probabilities and
    averages the resulting value over the whole dataset.

    With ``reduction = 'mean'``, this computes the entropy for each element of
        the ``input`` list.
    With ``reduction = None``, the sample-wise entropy is computed for each
        element of the ``input`` list.
    With ``reduction = 'correct vs. incorrect'``, the mean entropy is computed
        over the correct and incorrect samples, respectively.
    """

    def __init__(
            self,
            function: Callable[[Tensor,
                                Tensor,
                                Callable[[Tensor], Tensor],
                                bool], Tensor],
            len_data: int,
            dim: int = -1,
            reduction: Optional[str] = 'mean',
            is_classification: bool = True,
            assert_mean: bool = True):
        """FunctionalMetric initializer.

        Args:
            function: A function that maps the input and target tensor to
                a single scalar reduced to the sum
            len_data: The length of the dataset
            dim: The index of the input that should be used to compute
                the metric
            reduction: The reduction to apply to the metric value. Possible
                reductions: mean, correct vs. incorrect, None
            is_classification: Whether the task is a classification task
            assert_mean: Whether the input is already averaged over different runs
        """
        super().__init__()
        self.function = function
        self.len_data = len_data
        self.value = [] if reduction is None else 0.
        self.dim = dim
        self.is_classification = is_classification
        self.reduction = reduction
        self.assert_mean = assert_mean

        if reduction == 'mean':
            self.reduction_fn: Callable[[Tensor], Tensor] = torch.sum
        elif reduction == 'correct vs. incorrect':
            self.reduction_fn: Callable[[Tensor], Tensor] = lambda x, **kwargs: x
            self.n_correct = 0
        elif reduction is None:
            self.reduction_fn: Callable[[Tensor], Tensor] = lambda x, **kwargs: x.tolist()
        else:
            raise ValueError(f'Unknown reduction: {reduction}')

    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of input and target.

        Args:
            input: The class probabilities as a list of tensors
                 each shape [B, C] if self.assert_mean else of shape [B, N, C]
            target: (shape [B]) The target values
        """
        input_dim = input[self.dim]
        new_value = self.function(
            input_dim, target,
            reduction_fn=self.reduction_fn,
            classification=self.is_classification)
        if self.reduction == 'correct vs. incorrect':
            in_mean = input_dim if input_dim.ndim == 2 else input_dim.mean(dim=1)
            correct_inds: Tensor = torch.argmax(in_mean, dim=-1) == target
            self.n_correct += torch.sum(correct_inds).item()

            correct = new_value[correct_inds].sum()
            incorrect = new_value[~correct_inds].sum()

            new_value = torch.stack([correct, incorrect])
        self.value = self.value + new_value

    def summarize(self) -> Union[float, List[float], Dict[str, float]]:
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        if self.reduction == 'mean':
            return self.value.item() / self.len_data
        elif self.reduction == 'correct vs. incorrect':
            return {
                'correct': np.divide(self.value[0].item(), self.n_correct),
                'incorrect': np.divide(self.value[1].item(), self.len_data - self.n_correct)
            }
        elif self.reduction is None:
            return self.value


class Calibration(Metric):
    """The calibration metric.

    This metric computes the expected calibration error and for each bin the
    difference of accuracy and confidence, the accuracy and the confidence.
    """

    assert_mean = True

    def __init__(
            self,
            len_data: int,
            dim: int = -1,
            is_classification: bool = True,
            num_bins: int = 10):
        """Calibration initializer.

        Args:
            len_data: The length of the dataset
            dim: The index of the probabilities that should be used to compute
                the metric
            is_classification: Whether the task is a classification task
            num_bins: The number of bins into which the probabilities are
                discretized
        """
        super().__init__()
        self.is_classification = is_classification
        if is_classification:
            self.len_data = len_data

            self.dim = dim

            self.binned_accuracies = torch.zeros(num_bins)
            self.binned_confidences = torch.zeros(num_bins)
            self.binned_num_samples = torch.zeros(num_bins)

            bins = torch.linspace(0, 1, num_bins + 1)
            self.lower_bins = bins[:-1]
            self.upper_bins = bins[1:]

    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of class probabilities and
        target.

        Args:
            input: The class probabilities as a list of (shape [B, C])
                tensors
            target: (shape [B]) The target values
        """
        if self.is_classification:
            device = target.device

            self.lower_bins, self.upper_bins = self.lower_bins.to(device), self.upper_bins.to(device)

            confidences = input[self.dim].max(dim=1)[0]
            correct = (input[self.dim].argmax(dim=1) == target).float()
            inds = torch.logical_and(
                confidences[:, None] > self.lower_bins[None, :],
                confidences[:, None] <= self.upper_bins[None, :]).float()

            self.binned_accuracies = self.binned_accuracies.to(device) + correct @ inds
            self.binned_confidences = self.binned_confidences.to(device) + confidences @ inds
            self.binned_num_samples = self.binned_num_samples.to(device) + inds.sum(dim=0)

    def summarize(self) -> Tuple[float, List[float], List[float], List[float]]:
        """Summarizes the metric value after all batches were observed.

        Returns:
            expected calibration error
            difference of confidence and accuracy in each bin
            accuracy in each bin
            confidence in each bin
        """
        if self.is_classification:
            self.binned_accuracies /= (self.binned_num_samples + torch.finfo().tiny)
            self.binned_confidences /= (self.binned_num_samples + torch.finfo().tiny)
            aces = self.binned_confidences - self.binned_accuracies
            ece = (aces.abs() * self.binned_num_samples).nansum() / self.len_data
            return ece.item(), aces.tolist(), \
                self.binned_accuracies.tolist(), self.binned_confidences.tolist()
        else:
            return -1, [], [], []


class ApplyFunctionalMetric(Metric):
    """The metric that applies the FunctionalMetric the each element of the
    input list."""

    def __init__(
            self,
            function: Callable[[Tensor,
                                Tensor,
                                Callable[[Tensor], Tensor]], Tensor],
            len_data: int,
            reduction: Optional[str] = 'mean',
            is_classification: List[bool] = None,
            assert_mean: bool = True):
        """ApplyFunctionalMetric initializer.

        Args:
            function: A function that maps the input and target tensor to
                a single scalar reduced to the sum
            len_data: The length of the dataset
            reduction: The reduction to apply to the metric value. Possible
                reductions: mean, correct vs. incorrect, None
            is_classification: A list of boolean values indicating whether the metric is
                a classification metric or a regression metric
            assert_mean: Whether the input is already averaged over different runs
        """
        super().__init__()
        self.assert_mean = assert_mean
        self.functional_metrics = [FunctionalMetric(function, len_data, i, reduction, is_classification[i])
                                   for i in range(len(is_classification))]

    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of class probabilities and
        target.

        Args:
            input: The input as a list of (shape [B, C])
                tensors
            target: (shape [B]) The target values
        """
        for f in self.functional_metrics:
            f.update(input, target)

    def summarize(self) -> List[float]:
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        return [f.summarize() for f in self.functional_metrics]


class Entropy(ApplyFunctionalMetric):
    """The entropy metric."""

    assert_mean = True

    def __init__(
            self,
            len_data: int,
            reduction: Optional[str] = 'mean',
            is_classification: List[bool] = None):
        """Entropy initializer.

        Args:
            len_data: The length of the dataset
            reduction: The reduction to apply to the metric value. Possible
                reductions: mean, correct vs. incorrect, None
            is_classification: A list of boolean values indicating whether the metric is
                a classification metric or a regression metric
        """
        super().__init__(entropy, len_data, reduction, is_classification)


class StandardDeviation(ApplyFunctionalMetric):
    assert_mean = False

    def __init__(
            self,
            len_data: int,
            reduction: Optional[str] = 'mean',
            is_classification: List[bool] = None):
        """Standard deviation initializer.

        Args:
            len_data: The length of the dataset
            reduction: The reduction to apply to the metric value. Possible
                reductions: mean, None
            is_classification: A list of booleans indicating whether the metric
                is a classification metric or a regression metric
        """
        super().__init__(standard_deviation, len_data, reduction, is_classification)


class MetricSet:
    """Set of different metrics."""

    all_metrics = {'accuracy', 'brier score', 'negative log likelihood',
                   'calibration', 'entropy', 'entropy histogram',
                   'entropy correct vs. incorrect', 'mse', 'mae', 'std',
                   'std histogram', 'std correct vs. incorrect'}

    @staticmethod
    def metric_to_class(
            metric: str,
            len_data: int,
            dim: int,
            is_classification: List[bool]) -> Metric:
        """Converts a metric name to a metric class.

        Args:
            metric: The name of the metric
            len_data: The length of the dataset
            dim: The index of the probabilities that should be used to compute
                the metric (except entropy which is computed over the full list)
            is_classification: A list of booleans indicating whether the metric
                is a classification metric or a regression metric

        Returns:
            The metric class
        """
        dim_classification = is_classification[dim]
        if metric == 'accuracy':
            return FunctionalMetric(accuracy, len_data, dim, 'mean', dim_classification)
        elif metric == 'brier score':
            return FunctionalMetric(brier_score, len_data, dim, 'mean', dim_classification)
        elif metric == 'negative log likelihood':
            return FunctionalMetric(negative_log_likelihood, len_data, dim, 'mean', dim_classification)
        elif metric == 'calibration':
            return Calibration(len_data, dim, dim_classification)
        elif metric == 'entropy':
            return Entropy(len_data, 'mean', is_classification)
        elif metric == 'entropy histogram':
            return Entropy(len_data, None, is_classification)
        elif metric == 'entropy correct vs. incorrect':
            return Entropy(len_data, 'correct vs. incorrect', is_classification)
        elif metric == 'mse':
            return FunctionalMetric(squared_error, len_data, dim, 'mean', dim_classification)
        elif metric == 'mae':
            return FunctionalMetric(absolute_error, len_data, dim, 'mean', dim_classification)
        elif metric == 'std':
            return StandardDeviation(len_data, 'mean', is_classification)
        elif metric == 'std histogram':
            return StandardDeviation(len_data, None, is_classification)
        elif metric == 'std correct vs. incorrect':
            return StandardDeviation(len_data, 'correct vs. incorrect', is_classification)

    def __init__(
            self,
            len_data: int,
            metrics: Union[str, Iterable[str]] = None,
            dim: int = -1,
            is_classification: List[bool] = None):
        """MetricSet initializer.

        Args:
            len_data: The length of the dataset
            metrics: Either 'all' or a list of metrics strings
                (Possible values: 'accuracy', 'brier score',
                'negative log likelihood', 'entropy', 'calibration',
                'entropy histogram')
            dim: The index of the probabilities that should be used to compute
                the metric (except entropy which is computed over the full list)
            is_classification: A list of booleans indicating whether the metric
                is a classification metric or a regression metric
        """
        if metrics == 'all':
            metrics = self.all_metrics
        elif metrics is None:
            metrics = set()
        elif isinstance(metrics, str):
            metrics = {metrics}
        elif not isinstance(metrics, set):
            metrics = set(metrics)
        metrics = list(metrics & self.all_metrics)
        self.metric_classes = {metric: self.metric_to_class(metric, len_data, dim, is_classification)
                               for metric in metrics}
        self.is_classification = is_classification

    def update(self, logits: Union[List[Tensor], Tensor], target: Tensor):
        """Updates the metrics with a new batch of class logits and
        target.

        Args:
            logits: The class logits as a list of (shape [B, C]) tensors or a
                single (shape [B, C]) tensor
            target: (shape [B]) The target values
        """
        if self.metric_classes:
            if not isinstance(logits, list):
                logits = [logits]

            for logit in logits:
                if not isinstance(logit, Tensor):
                    raise TypeError(
                        f'Expected logits to be a list of Tensors, '
                        f'got {type(logit)}')
                if logit.ndim == 2:
                    logit.unsqueeze_(1)

            input = [F.softmax(logit, dim=-1) if is_classification else logit
                     for is_classification, logit in zip(self.is_classification, logits)]

            mean = [i.mean(dim=1) for i in input]

            for metric_class in self.metric_classes.values():
                if metric_class.assert_mean:
                    metric_class.update(mean, target)
                else:
                    metric_class.update(input, target)

    def summarize(self) -> Dict[str, Any]:
        """Summarizes the metrics values after all batches were observed.

        Returns:
            A dict containing the summerized values
        """
        return {metric: metric_class.summarize()
                for metric, metric_class in self.metric_classes.items()}


def run_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: Callable[[Tensor, Tensor], Tensor],
        is_classification: Union[bool, List[bool]] = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train: bool = False,
        metrics: Union[str, List[str]] = None,
        metrics_dim: int = -1,
        prefix: str = '',
        return_if_loss_nan_or_inf: bool = False) \
        -> Dict[str, Union[float, np.array]]:
    """Runs one epoch over the dataloader.

    If ``train`` is True, the model is optimized, else the model is evaluated.

    Args:
        model: A PyTorch model
        dataloader: A PyTorch dataloader
        criterion: A function mapping the logits and targets to the loss
        is_classification: A list of booleans indicating whether the metric
            is a classification metric or a regression metric (or a single
            boolean if all metrics are of the same type)
        optimizer: A PyTorch optimizer
        train: Whether the model is trained or evaluated
        metrics: Either 'all' or a list of metrics strings
            (Possible values: 'accuracy', 'brier score',
            'negative log likelihood', 'entropy', 'calibration',
            'entropy histogram')
        metrics_dim: The index of the logits that should be used to compute the
            loss
        prefix: The description of the tqdm bar
        return_if_loss_nan_or_inf: Whether to return if the loss is NaN or Inf

    Returns:
        The metrics over the epoch
    """
    if train:
        model.train()
    else:
        model.eval()

    if not isinstance(is_classification, list):
        is_classification = [is_classification]

    metrics_obj = MetricSet(
        len(dataloader.dataset), metrics, metrics_dim,
        is_classification)

    mean_loss = torch.as_tensor(0., device=device)

    model.to(device)
    with tqdm(dataloader, file=sys.stdout) as t:
        t.set_description(prefix)
        for step, (features, targets) in enumerate(t):
            features, targets = features.to(device), targets.to(device)
            logits = model(features)

            if not isinstance(logits, list):
                logits = [logits]

            for logit in logits:
                if not isinstance(logit, Tensor):
                    raise TypeError(
                        f'Expected logits to be a list of Tensors, '
                        f'got {type(logit)}')
                if logit.ndim == 2:
                    logit.unsqueeze_(1)

            if is_classification[metrics_dim]:
                logit = logits[metrics_dim]
                if logit.shape[1] == 1:
                    loss = criterion(logit[:, 0, :], targets)
                else:
                    probs = F.softmax(logits[metrics_dim], dim=-1)
                    loss = criterion(probs.mean(dim=1).log(), targets)
            else:
                loss = criterion(logits[metrics_dim].mean(dim=1), targets)
            if loss.isnan() or loss.isinf():
                if return_if_loss_nan_or_inf:
                    return {'loss': loss.item()}
                elif loss.isnan():
                    print(f'Loss is nan in step {step}. This step is ignored for training.')
            if not loss.isnan() and train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss += loss.detach() / len(dataloader)
            metrics_obj.update([logit.detach() for logit in logits], targets.detach())
            t.set_postfix(loss=loss.detach().item())

    metrics_dict = metrics_obj.summarize()

    metrics_dict['loss'] = mean_loss.item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics_dict


def get_verbose_string(d):
    """Creates the string that summarizes the dictionary of metrics.

    Args:
        d: The dictionary of metrics

    Returns:
        A string containing the metrics
    """
    assert 'loss' in d.keys()
    verbose_string = f'loss: {d["loss"]: .5f}'
    for key, value in d.items():
        if key != 'loss':
            verbose_string += f', {key}: {value:.2f}'
            if key == 'accuracy':
                verbose_string += '%'
    return verbose_string


def fit(
        model: nn.Module,
        dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader],
        criterion: Callable[[Tensor, Tensor], Tensor],
        weight_decay: float,
        is_classification: Union[bool, List[bool]] = True,
        learning_rate: float = 2e-3,
        use_validation_set: bool = True,
        num_epochs: int = 100,
        patience: int = 10,
        metrics_run_epoch: Union[str, List[str]] = None) \
        -> Dict[str, List[Dict[str, float]]]:
    """Runs the full optimization routine for a model.

    Trains for ``num_epochs`` epochs with the Adam optimizer and the
    ReduceLROnPlateau learning rate scheduler and evaluates the model after
    every epoch on the validation set and test set. The test set metrics are NOT
    used for the training.

    The early stopping is determined either based on the train or validation
    loss dependent on ``use_validation_set``.

    Args:
        model: A PyTorch model
        dataloader:  A tuple of train-, val-, and test-dataloaders
        criterion: A function mapping the logits and targets to the loss
        weight_decay: The weight decay used in the optimizer
        is_classification: A list of booleans indicating whether the metric
            is a classification metric or a regression metric (or a single
            boolean if all metrics are of the same type)
        learning_rate: The learning rate used in the Adam optimizer
        use_validation_set: Whether the training or validation set should be
            used for early stopping
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        metrics_run_epoch: Either 'all' or a list of metrics strings
            (Possible values: 'accuracy', 'brier score',
            'negative log likelihood', 'entropy', 'calibration',
            'entropy histogram')

    Returns:
        A dict containing the training, validation and test metrics
    """
    if metrics_run_epoch is None:
        metrics_run_epoch = ['accuracy']
    dataloader_train, dataloader_val, dataloader_test = dataloader

    # variables for early stopping
    best_model_state_dict = model.state_dict()
    metric_for_best_model = 'loss'
    metric_should_be_large = False
    early_stopping_coefficient = 1 if metric_should_be_large else -1
    best_value = -float('Inf') * early_stopping_coefficient
    steps_without_improvement = 0

    metrics = {
        'train': [],
        'test': [],
    }
    if dataloader_val is not None and use_validation_set:
        metrics['val'] = []

    criterion.to(device)
    parameter_container = criterion if hasattr(criterion, 'model') else model
    params = [p for p in parameter_container.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    lr_schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, verbose=True, patience=5),
    ]

    for epoch in range(num_epochs):
        string_length = str(len(str(num_epochs - 1)))
        prefix = ('Epoch {epoch:' + string_length + 'd}: ').format(epoch=epoch + 1)
        train_metrics = run_epoch(
            model, dataloader_train, criterion, optimizer=optimizer,
            train=True, metrics=metrics_run_epoch, prefix=prefix + 'Train',
            is_classification=is_classification)
        metrics['train'].append(train_metrics)

        verbose_string = prefix
        verbose_string += f'Train: {get_verbose_string(train_metrics)} | '

        if dataloader_val is not None and use_validation_set:
            val_metrics = run_epoch(
                model, dataloader_val, criterion,
                train=False, metrics=metrics_run_epoch, prefix=prefix + 'Val',
                is_classification=is_classification)
            metrics['val'].append(val_metrics)
            current_value = val_metrics[metric_for_best_model]
            lr_scheduler_metric = val_metrics['loss']

            verbose_string += f'Val: {get_verbose_string(val_metrics)} | '
        else:
            current_value = train_metrics[metric_for_best_model]
            lr_scheduler_metric = train_metrics['loss']
        for lr_scheduler in lr_schedulers:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(lr_scheduler_metric)
            else:
                lr_scheduler.step()

        test_metrics = run_epoch(
            model, dataloader_test, criterion,
            train=False, metrics=metrics_run_epoch, prefix=prefix + 'Test',
            is_classification=is_classification)
        metrics['test'].append(test_metrics)

        verbose_string += f'Test: {get_verbose_string(test_metrics)}'

        tqdm.write(verbose_string)

        if early_stopping_coefficient * current_value > early_stopping_coefficient * best_value:
            best_value = current_value
            best_model_state_dict = model.state_dict()
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break

    model.load_state_dict(best_model_state_dict)
    return metrics


def compute_curvature(
        model: nn.Module,
        curvs: List[Curvature],
        dataloader: DataLoader,
        return_data_log_likelihood: bool = True,
        num_samples: int = 1,
        invert: bool = False,
        make_positive_definite: bool = False,
        device: torch.device = device,
        seed: int = seed,
        categorical: bool = True,
        save_path: str = None,
        save_every_n_steps: int = 100):
    """Computes the curvatures.

    Changes the curvatures and can return the negative log-likelihood.

    Args:
        model: A PyTorch model
        curvs: A list of curvature objects corresponding to ``model``
        dataloader: A PyTorch dataloader
        return_data_log_likelihood: Whether the negative log-likelihood should
            be computed and returned
        num_samples: The number of samples used to compute the curvature
        invert: Whether the curvature should be inverted afterwards
        make_positive_definite: Whether the curvature should be made positive
            definite
        device: A device
        seed: A seed
        categorical: Whether the targets are categorical or continuous
        save_path: Path where intermediate and final results of the curvature
            should be saved (or not saved at all if ``save_path = None``
        save_every_n_steps: The frequency of saving the curvature

    Returns:
        negative log-likelihood if ``return_data_log_likelihood`` else None
    """
    model.to(device)
    model.train()
    generator = torch.Generator('cpu').manual_seed(seed)
    criterion = nn.CrossEntropyLoss() if categorical else HalfMSELoss()
    log_likelihood_criterion = nn.CrossEntropyLoss(reduction='sum') if categorical \
        else HalfMSELoss(reduction='sum')
    log_likelihood = 0.
    for curv in curvs:
        curv.model.train()
        curv.model.requires_grad_(True)
    for i, (features, targets) in enumerate(tqdm(dataloader, desc='Computing the curvature: ', file=sys.stdout)):
        if save_path is not None and i % save_every_n_steps == 0:
            torch.save([curv.state_dict() for curv in curvs], save_path)
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        if isinstance(logits, list):
            logits = logits[-1]
        if logits.ndim == 3:
            if categorical:
                if logits.shape[1] > 1:
                    probs = F.softmax(logits, dim=-1)
                    logits = probs.mean(dim=1).log()
                else:
                    logits = logits[:, 0, :]
            else:
                logits = logits.mean(dim=1)
        if return_data_log_likelihood:
            log_likelihood += log_likelihood_criterion(logits, targets).detach().item()
        for _ in range(num_samples):
            generator_local = torch.Generator(device) \
                .manual_seed(torch.randint(0, 0xffff_ffff, [], generator=generator).item())
            sampled_labels = torch.multinomial(F.softmax(logits, dim=1), 1, True, generator=generator_local)[:, 0] \
                if categorical else torch.normal(logits, 1., generator=generator_local)

            loss = criterion(logits, sampled_labels)
            model.zero_grad()
            loss.backward(retain_graph=num_samples > 1)

            for curv in curvs:
                curv.update(batch_size=features.size(0))
    for curv in curvs:
        curv.remove_hooks_and_records()
        curv.scale(num_batches=len(dataloader) * num_samples, make_pd=make_positive_definite)
        if invert:
            curv.invert()
    if save_path is not None:
        torch.save([curv.state_dict() for curv in curvs], save_path)
    if return_data_log_likelihood:
        return log_likelihood


def optimize(
        params: List[Tensor],
        flexible_params: List[Tensor],
        objective: Callable[[List[Tensor]], Tensor],
        max_iter: int = 500,
        eps: float = 1e-6,
        patiance: int = 10,
        retain_graph: bool = False):
    """Minimizes parameters with respect to a differentiable objective.

    The Adam optimizer is used in combination with an exponential learning rate
    scheduler. The optimization runs for ``max_iter`` steps or stops if all
    derivative of the parameters is smaller than ``eps``.

    Args:
        params: A list of parameter tensors
        flexible_params: A list of parameter tensors that can be changed during
            the optimization
        objective: A function mapping the parameters on a value
        max_iter: The maximal number of iterations
        eps: The optimization is stopped if all gradients are smaller than
        this value
        patiance: The optimization is stopped if no improvement is made for
        ``patiance`` steps
        retain_graph: Whether the computational graph should be retained
    """
    best_params = [param.clone() for param in params]
    best_value = torch.inf
    steps_without_improvement = 0

    for param in flexible_params:
        param.requires_grad = True

    optimizer = torch.optim.Adam(flexible_params, .5)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1 - 1e-6)

    with tqdm(range(max_iter), file=sys.stdout) as t:
        for _ in t:
            loss = objective(*params)

            if loss < best_value - eps:
                best_value = loss.item()
                best_params = [param.clone().detach() for param in params]
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patiance:
                    break

            t.set_postfix(best_value=best_value)
            # if i % 10 == 0:
            #     print(f'Iteration {i}: loss {loss.item()}')
            optimizer.zero_grad()
            loss.backward(retain_graph=retain_graph)
            for param in flexible_params:
                param.grad.nan_to_num_(0.)
            optimizer.step()
            lr_scheduler.step()
            # print([param.grad for param in params])
            if all((param.grad.abs() < eps).all() for param in flexible_params):
                break
    return best_params


def mc_allester_bound(
        expected_empirical_risk: Union[Tensor, float],
        kl_divergence: Union[Tensor, float],
        len_data: int,
        confidence: float = .8):
    r"""Computes the McAllester bound.

    .. math::
        \text{eer} + \sqrt{\frac{\text{kl} + \ln{\frac{2\sqrt{N}}{\varepsilon}}}{2 N}}

    Args:
        expected_empirical_risk: The expected empirical risk over the training
            data (:math:`\text{eer}`)
        kl_divergence: The Kullback-Leibler-divergence of the posterior with the
            prior (:math:`\text{kl}`)
        len_data: The length of the training dataset (:math:`N`)
        confidence: The bound of the probability (:math:`\varepsilon`)

    Returns:
        A tensor containing the McAllester bound value
    """
    reg = (kl_divergence + log(2 * sqrt(len_data)) - log(1 - confidence)) / (2. * len_data)
    return expected_empirical_risk + torch.sqrt(reg)


def catoni_bound(
        expected_empirical_risk: Union[Tensor, float],
        kl_divergence: Union[Tensor, float],
        len_data: int,
        log_catoni_scale: Optional[Union[Tensor, float]] = None,
        confidence: float = .8,
        return_log_catoni_scale: bool = False,
        max_iter: int = 100_000):
    r"""Computes the Catoni bound.

    .. math::
        \inf_{c > 0} \frac{1 - \exp(-c \text{eer} - \frac{\text{kl} - \ln{\varepsilon}}{N})}{1 - \exp(-c)}

    The minimal catoni scale is computed with max_iter steps if it is not given.

    Args:
        expected_empirical_risk: The expected empirical risk over the training
            data (:math:`\text{eer}`)
        kl_divergence: The Kullback-Leibler-divergence of the posterior with the
            prior (:math:`\text{kl}`)
        len_data: The length of the training dataset (:math:`N`)
        log_catoni_scale: The logarithm of the catoni scale (:math:`\log{c}`).
            If `None`` the catoni scale is optimized.
        confidence: The bound of the probability (:math:`\varepsilon`)
        return_log_catoni_scale: If the ``log_catoni_scale`` should be returned
            as second argument
        max_iter: The maximal number of iterations to obtain the catoni scale

    Returns:
        A tensor containing the Catoni bound value and
        if ``return_log_catoni_scale`` also the ``log_catoni_scale``
    """
    if log_catoni_scale is None:
        log_catoni_scale = torch.zeros([], device=device)
        objective = lambda x: catoni_bound(
            expected_empirical_risk=expected_empirical_risk, kl_divergence=kl_divergence,
            len_data=len_data, log_catoni_scale=x, confidence=confidence)
        log_catoni_scale = \
            optimize([log_catoni_scale], [log_catoni_scale], objective, max_iter=max_iter, retain_graph=True)[0]

    reg = (kl_divergence - log(1 - confidence)) / len_data
    bound = (1. - torch.exp(- log_catoni_scale.exp() * expected_empirical_risk - reg)) / (
            1. - torch.exp(- log_catoni_scale.exp()))

    if return_log_catoni_scale:
        return bound, log_catoni_scale
    else:
        return bound


def get_dataset_and_name(dataloader: DataLoader):
    """Returns the dataset of the dataloader and its name.

    Args:
        dataloader: A PyTorch dataloader

    Returns:
        The dataset and the name
    """
    dataset = dataloader.dataset
    ds = dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return dataset, type(ds).__name__


class HalfMSELoss(nn.MSELoss):
    """A loss function that uses the mean squared error with half the weight of
    the mean squared error.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction) / 2.
