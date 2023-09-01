"""Official implementation of Bayesian Progressive Neural Networks from
`Learning Expressive Priors for Generalization and Uncertainty Estimation in
Neural Networks <https://proceedings.mlr.press/v202/schnaus23a/schnaus23a.pdf>`_
"""

from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.curvature.curvatures import *
from .criterions import Criterion
from .curvature_scalings import CurvatureScaling
from .pnn import ProbabilisticProgressiveNeuralNetwork
from .utils import device, fit, \
    compute_curvature


class BayesianProgressiveNeuralNetwork(ProbabilisticProgressiveNeuralNetwork):
    """Bayesian Progressive Neural Networks (BPNN).

    This network architecture builds on top of Progressive Neural Networks and
    computes a posterior for the weighted layers. For this, it utilizes a prior
    computed on a different related dataset and computes the posterior with
    Laplace approximation.

    The prior for the main columns is the posterior of the prior task. The prior
    for the lateral connections is posterior of the outgoing layer.

    The steps of training a Bayesian Progressive Neural Network are
    1. Compute the posterior on a related dataset (init_prior)
    2. For each incoming task:
        1. Add a new column (add_new_column)
        2. Fit weights of last column and incoming lateral connections to the
            dataset
        3. Compute the curvature on the dataset (compute_new_curvature)
        4. Combine the prior curvature and new curvature of the likelihood with
            a curvature scaling to obtain the new posterior (add_new_posterior)
    """

    def __init__(
            self,
            prior: Curvature,
            backbone: nn.Module = None,
            last_layer_name: Optional[str] = None,
            lateral_connections: Optional[List[str]] = None,
            train_resample_slice: slice = slice(-1),
            train_num_samples: int = 1,
            eval_resample_slice: slice = slice(None),
            eval_num_samples: int = 100,
            weight_decay: float = 1e-5,
            weight_decay_layer_names: Optional[List[str]] = None,
            curvature_device: Optional[torch.device] = torch.device('cpu'),
            curvature_scaling_device: Optional[torch.device] = torch.device('cpu'),
            ):
        """BayesianProgressiveNeuralNetwork initializer.

        Args:
            prior: A Curvature (that is inverted) that is used as prior
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
            weight_decay: The weight decay used for the prior
            weight_decay_layer_names: Layers that use an isotropic Gaussian
                prior with default_weight_decay (should usually include
                last_layer_name when the columns have different number of
                outputs)
            curvature_device: The device for the curvature
            curvature_scaling_device: The device used for the curvature scaling
        """
        super().__init__(
            prior.model, backbone, last_layer_name, lateral_connections,
            train_resample_slice, train_num_samples, eval_resample_slice, eval_num_samples)
        prior.model_state = prior.model.state_dict()  # updates the names of the state dict
        self.weight_decay = weight_decay
        self.prior = prior

        # approximated posterior curvatures (are needed for sampling and KL-divergence computation)
        self.posterior: List[List[Curvature]] = []

        self.scales = {}
        self.temperature_scaling = {}

        if weight_decay_layer_names is None and last_layer_name is not None:
            weight_decay_layer_names = [last_layer_name]
        for name in set(weight_decay_layer_names).intersection(self.lateral_connections):
            weight_decay_layer_names[weight_decay_layer_names.index(name)] = f'{name}.0'
        self.weight_decay_layer_names = weight_decay_layer_names

        self.curvature_device = curvature_device
        self.curvature_scaling_device = curvature_scaling_device

        self.fixed_model_size_trace = 0.

    @property
    def previous_tasks(self):
        """The number of previously finished tasks.

        Different from Progressive Neural Networks as the task is only finished
        when the posterior is available.

        Returns:
            The number of previously finished tasks.
        """
        return len(self.posterior) + 1

    def full_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the whole state.

        Returns:
            A dictionary containing the whole state
        """
        full_state_dict = super().full_state_dict()

        full_state_dict.update(
            {
                'posterior': [
                    [curvature_column.state_dict() for curvature_column in curvature]
                    for curvature in self.posterior
                ] if self.posterior is not None else None,
                'prior': self.prior.state_dict() if self.prior is not None else None,
                'weight_decay': self.weight_decay,
                'scales':
                    [
                        [
                            {name: self.scales[module] for name, module in column.named_modules()
                             if module in self.scales.keys()}
                            for column in network
                        ]
                        for network in self.networks
                    ],
                'temperature_scaling':
                    [
                        [
                            {name: self.temperature_scaling[module] for name, module in column.named_modules()
                             if module in self.temperature_scaling.keys()}
                            for column in network
                        ]
                        for network in self.networks
                    ],
                'weight_decay_layer_names': self.weight_decay_layer_names,
                'fixed_model_size_trace': self.fixed_model_size_trace
            }
        )
        return full_state_dict

    def load_full_state_dict(self, full_state_dict):
        """Copies the whole state from full_state_dict.

        Args:
            full_state_dict: A dict containing the full state
        """
        super().load_full_state_dict(full_state_dict)
        self.weight_decay = full_state_dict['weight_decay']
        self.weight_decay_layer_names = full_state_dict['weight_decay_layer_names']

        self.prior = eval(full_state_dict['prior']['class'])(self.prior.model)
        self.prior.remove_hooks_and_records()
        self.prior.load_state_dict(full_state_dict['prior'])

        self.fixed_model_size_trace = full_state_dict['fixed_model_size_trace']

        if full_state_dict['networks']:
            if full_state_dict['posterior'] is not None:
                self.posterior = []
                for curvature_states, network in zip(full_state_dict['posterior'], self.networks):
                    posterior = []
                    for curvature_state, column in zip(curvature_states, network):
                        curvature_column = eval(curvature_state['class'])(column)
                        curvature_column.remove_hooks_and_records()
                        curvature_column.load_state_dict(curvature_state)
                        posterior.append(curvature_column)
                    self.posterior.append(posterior)

            for scale_network, network in zip(full_state_dict['scales'], self.networks):
                for scale_column, column in zip(scale_network, network):
                    for name, module in column.named_modules():
                        if name in scale_column.keys():
                            self.scales[module] = scale_column[name]

            for temperature_network, network in zip(full_state_dict['temperature_scaling'], self.networks):
                for temperature_column, column in zip(temperature_network, network):
                    for name, module in column.named_modules():
                        if name in temperature_column.keys():
                            self.temperature_scaling[module] = temperature_column[name]

            self.networks[-1].requires_grad_(True)

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
                completely (new random initialization)
        """
        for posterior_list in self.posterior:
            for posterior in posterior_list:
                posterior.model.load_state_dict(posterior.model_state)
        super().add_new_column(is_classification, output_size, differ_from_previous, resample_base_network)

    def update_scaling(
            self,
            alpha: Optional[Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            beta: Optional[Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            temperature_scaling: Optional[
                Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            update_slice: Optional[slice] = slice(-1, None)):
        """Updates the scales and temperature scaling.

        Args:
            alpha: The new value for the scale of the prior.
            beta: The new value for the scale of the likelihood.
            temperature_scaling: The new value for the temperature scaling.
            update_slice: The slice of the networks to be updated only used if the
                scalar is a number.
        """

        for i, scaling in enumerate([alpha, beta]):
            if scaling is not None:
                if isinstance(scaling, dict):
                    for module, scale in scaling.items():
                        if module not in self.scales.keys():
                            self.scales[module] = [None, None]
                        self.scales[module][i] = scale
                else:
                    for posterior_list in self.posterior[update_slice]:
                        for posterior in posterior_list:
                            for module in posterior.state.keys():
                                if module not in self.scales.keys():
                                    self.scales[module] = [None, None]
                                self.scales[module][i] = scaling

        if temperature_scaling is not None:
            if isinstance(temperature_scaling, dict):
                for module, scale in temperature_scaling.items():
                    self.temperature_scaling[module] = scale
            else:
                for posterior_list in self.posterior[update_slice]:
                    for posterior in posterior_list:
                        for module in posterior.state.keys():
                            self.temperature_scaling[module] = temperature_scaling

    def update_posterior(
            self,
            alpha: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            beta: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            curvature_list: List[Curvature]):
        """Updates the posterior.

        This method sets the posterior curvature to
            alpha * prior + beta * curvature.

        Args:
            alpha: The prior scale
            beta: The likelihood scale
            curvature_list: The list of curvatures to be used for the update
        """
        assert len(curvature_list) == len(self.posterior)

        priors = [posterior[-1] for posterior in self.posterior[:-1]] + [self.prior]

        self.posterior[-1] = [curvature.add_and_scale(
            prior,
            [beta, alpha],
            self.weight_decay_layer_names,
            self.weight_decay)
                              for curvature, prior in zip(curvature_list, priors)]

        for posterior in self.posterior[-1]:
            posterior.invert()

    def add_new_posterior(
            self,
            curvature_scaling: CurvatureScaling,
            dataloader: DataLoader,
            num_samples: int = 1,
            return_curvature: bool = False):
        """Adds a new posterior.

        Args:
            curvature_scaling: The curvature scaling used to determine
                alpha and beta
            dataloader: The dataloader that is used to compute the curvature
            num_samples: The number of samples used to compute the curvature
            return_curvature: Whether the curvature should be returned
        """
        assert len(self.networks) == len(self.posterior) + 1, \
            'compute_new_curvature and add_new_column should be called before add_new_posterior'

        new_curvature = [type(self.prior)(network, device=self.curvature_device) for network in
                         self.networks[-1]]
        negative_data_log_likelihood = compute_curvature(
            self, new_curvature, dataloader,
            num_samples=num_samples, invert=False,
            categorical=self.is_classification[-1])
        for curvature in new_curvature:
            curvature.remove_hooks_and_records()

        self.posterior.append([])

        def update(alpha, beta, temperature_scaling):
            self.update_posterior(alpha, beta, new_curvature)
            self.update_scaling(alpha, beta, temperature_scaling)

        trace = lambda temperature_scaling: self.trace(
            new_curvature,
            temperature_scaling=temperature_scaling)
        kl_divergence = lambda temperature_scaling: self.kl_divergence(temperature_scaling=temperature_scaling)

        modules = [module for curvature in new_curvature for module in curvature.state.keys()]

        for curvature in self.posterior[-1] + new_curvature:
            curvature.to(self.curvature_scaling_device)

        alpha, beta, temperature_scaling = curvature_scaling.find_optimal_scaling(
            update=update,
            model=self,
            trace=trace,
            kl_divergence=kl_divergence,
            modules=modules,
            len_data=len(dataloader.dataset),
            negative_data_log_likelihood=negative_data_log_likelihood,
            scaled_fixed_model_size=self.fixed_model_size_trace,
            device=self.curvature_scaling_device
        )

        for curvature in self.posterior[-1] + new_curvature:
            curvature.to(self.curvature_device)

        print(alpha, beta, temperature_scaling)

        self.update_posterior(alpha, beta, new_curvature)
        self.update_scaling(alpha=alpha, beta=beta, temperature_scaling=temperature_scaling)
        self.fixed_model_size_trace += self.trace(new_curvature).item()

        if return_curvature:
            return new_curvature, negative_data_log_likelihood

    def _sample_and_replace(self, resample_slice: slice):
        # Note that resample_slice is the slice of the posterior,
        # but there might be more columns than posteriors

        # implement sampling and replacing by the posterior
        for _, column in zip(
                self.networks[resample_slice],
                self.posterior):
            for curv in column:
                curv.sample_and_replace(temperature_scaling=self.temperature_scaling)

    def get_quadratic_term(self, penalty_slice=slice(-1, None)):
        """Computes the quadratic term from the normal prior and posterior.

        Let :math:`\\theta` be the parameters of the column and
        :math:`\\hat{\\theta}` be the parameters and :math:`\\Sigma` the
        curvature of the prior, then this method computes
         .. math:
            \frac{1}{\\tau}(\\theta - \\hat{\\theta})^T \\Sigma (\\theta - \\hat{\\theta})
        :math:`\\Sigma_l = weight_decay I` for all layers in weight_decay_layer_names.

        Args:
            penalty_slice: The column indices the quadratic term should be
                computed for

        Returns:
            A tensor containing the quadratic term
        """
        out = torch.as_tensor(0., device=device)
        for column in self.networks[penalty_slice]:
            for prior, network in zip(self.posterior, column[:-1]):
                out += prior[-1].get_quadratic_term(
                    network,
                    weight_decay_layer_names=self.weight_decay_layer_names,
                    default_weight_decay=self.weight_decay)
            out += self.prior.get_quadratic_term(
                column[-1],
                weight_decay_layer_names=self.weight_decay_layer_names,
                default_weight_decay=self.weight_decay)
        return out

    def kl_divergence(self, penalty_slice=slice(-1, None), temperature_scaling=None):
        """Computes the Kullback-Leibler(KL)-divergence.

        The KL-divergence is computes for the columns in penalty_slice and their
        corresponding priors.

        Args:
            penalty_slice: The column indices the KL-divergence should be
                computed for
            temperature_scaling: The temperature scaling

        Returns:
            A tensor containing the KL-divergence
        """
        assert len(self.posterior) == len(self.networks), \
            'posterior is not available for all networks'

        if temperature_scaling is None:
            temperature_scaling = self.temperature_scaling

        out = torch.as_tensor(0., device=device)
        for column in self.posterior[penalty_slice]:
            for prior, posterior in zip(self.posterior, column[:-1]):
                out += posterior.kl_divergence(prior[-1], temperature_scaling=temperature_scaling)
            out += column[-1].kl_divergence(
                self.prior,
                temperature_scaling=temperature_scaling,
                weight_decay_layer_names=self.weight_decay_layer_names,
                default_weight_decay=self.weight_decay)
        return out

    def trace(
            self,
            curvature_list: List[Curvature],
            temperature_scaling: Optional[
                Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None):
        """Computes the trace of the matrix product of the Fisher matrix in curvature_list
        and the last posterior.

        Args:
            curvature_list: A list of curvatures
            temperature_scaling: The temperature scaling

        Returns:
            A tensor containing the trace
        """
        assert len(curvature_list) == len(self.posterior)

        if temperature_scaling is None:
            temperature_scaling = self.temperature_scaling

        out = torch.as_tensor(0., device=device)
        for curvature, posterior in zip(curvature_list, self.posterior[-1]):
            out += curvature.trace_of_mm(posterior, temperature_scaling=temperature_scaling)
        return out


def init_prior(
        base_network: nn.Module,
        curvature_type: type,
        dataloader: DataLoader,
        weight_decay: float,
        curvature_scaling: CurvatureScaling,
        isotropic_prior: bool = False,
        curvature_device: Optional[torch.device] = None,
        curvature_scaling_device: Optional[torch.device] = None,
        curvature_path: str = None,
        negative_data_log_likelihood: float = None,
        len_data: int = None
        ) -> Curvature:
    """Initializes the prior.

    It computes the curvature around the base_network using the dataloader and
    returns a curvature containing the prior distribution. If curvature_path is
    provided, the curvature is loaded from the path (also requires the
    negative_data_log_likelihood and len_data). If isotropic_prior is True, the
    prior is an isotropic Gaussian with weight_decay as variance.

    If the curvature is computed and not isotropic, we scale it using the
    curvature_scaling.

    Args:
        base_network: The base network
        curvature_type: The class of curvature to be used
        dataloader: The dataloader to compute the curvature
        weight_decay: The weight decay used for the prior
        curvature_scaling: The curvature scaling used to scale the prior
        isotropic_prior: Whether the prior should be isotropic
        curvature_device: The device used for the curvature
        curvature_scaling_device: The device used for the curvature scaling
        curvature_path: The path to the curvature
        negative_data_log_likelihood: The negative data log likelihood of the
            curvature
        len_data: The number of data points used to compute the curvature

    Returns:
        The prior
    """
    if curvature_device is None:
        curvature_device = device
    prior_curvature = curvature_type(base_network, device=curvature_device)

    if curvature_path and negative_data_log_likelihood and len_data:

        curvature_state_dict = torch.load(curvature_path, map_location=curvature_device)
        prior_curvature.load_state_dict(curvature_state_dict)
    else:
        if isotropic_prior:
            negative_data_log_likelihood = compute_curvature(
                model=base_network,
                dataloader=[next(iter(dataloader))],
                curvs=[prior_curvature],
                return_data_log_likelihood=True,
                num_samples=1,
                invert=True,
                make_positive_definite=False,
                device=curvature_device,
                categorical=True,
            )
        else:
            negative_data_log_likelihood = compute_curvature(
                model=base_network,
                dataloader=dataloader,
                curvs=[prior_curvature],
                return_data_log_likelihood=True,
                num_samples=1,
                invert=True,
                make_positive_definite=False,
                device=curvature_device,
                categorical=True,
            )
            len_data = len(dataloader.dataset)
    prior_curvature.remove_hooks_and_records()

    isotropic = prior_curvature.eye_like(weight_decay=weight_decay)
    isotropic.remove_hooks_and_records()

    if isotropic_prior:
        return isotropic

    for curvature in [isotropic, prior_curvature]:
        curvature.to(curvature_scaling_device)

    def update(alpha, beta, temperature_scaling):
        global prior
        prior = prior_curvature.add_and_scale(isotropic, [beta, alpha])
        prior.invert()

    def trace(temperature_scaling):
        global prior
        return prior_curvature.trace_of_mm(
            prior,
            temperature_scaling=temperature_scaling
        )

    def kl_divergence(temperature_scaling):
        global prior
        return prior.kl_divergence(
            isotropic,
            temperature_scaling=temperature_scaling
        )

    alpha, beta, temperature_scaling = curvature_scaling.find_optimal_scaling(
        update=update,
        model=base_network,
        trace=trace,
        kl_divergence=kl_divergence,
        modules=prior_curvature.state.keys(),
        len_data=len_data,
        negative_data_log_likelihood=negative_data_log_likelihood,
        scaled_fixed_model_size=0,
        device=curvature_scaling_device
    )
    update(alpha, beta, temperature_scaling)
    print(alpha, beta, temperature_scaling)

    prior.scale_inverse(temperature_scaling)

    for curvature in [isotropic, prior_curvature, prior]:
        curvature.to(curvature_device)

    base_network.load_state_dict(prior.model_state)
    return prior


def dataset_step_bpnn(
        model: BayesianProgressiveNeuralNetwork,
        dataloader: Tuple[DataLoader, DataLoader, DataLoader],
        loss_function: nn.Module,
        output_size: int,
        weight_decay: float,
        learning_rate: float,
        num_epochs: int,
        patience: int,
        train_dataset: Dataset,
        curvature_scaling: CurvatureScaling,
        criterion: Criterion,
        curvature_num_samples: int,
        **kwargs) -> Dict[str, List[Dict[str, float]]]:
    """The dataset step of BPNN for fit_pnn.

    A new column is added and fitted. Moreover, the curvature is computed and
    the posterior is set.

    Args:
        model: A BayesianProgressiveNeuralNetwork object
        dataloader: A tuple of train-, val-, and test-dataloaders
        loss_function: The loss function to use
        output_size: The dimension of the network output
        weight_decay: The weight decay used in the BPNN
        learning_rate: The learning rate used to train the column
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        train_dataset: The training dataset (usually dataloader.dataset)
        curvature_scaling: The curvature scaling to scale the posterior
        criterion: The criterion that is used to optimize the parameters
        curvature_num_samples: The number of samples used to compute the curvature

    Returns:
        The metrics computed during the training
    """
    criterion.len_data = torch.as_tensor(len(train_dataset))
    criterion.loss_function = loss_function
    is_classification = type(loss_function) is nn.CrossEntropyLoss

    # add new column
    if len(model.networks) == len(model.posterior):
        model.add_new_column(
            is_classification=is_classification,
            output_size=output_size)
        eval_num_samples = model.eval_num_samples
        model.eval_num_samples = 1
        train_metrics = fit(
            model, dataloader, criterion, weight_decay=.0,
            is_classification=model.is_classification,
            learning_rate=learning_rate,
            num_epochs=num_epochs, patience=patience)
        model.eval_num_samples = eval_num_samples
    else:
        train_metrics = {}

    # add new posterior
    curvature_scaling.reset(
        dataloader=dataloader[1],
        criterion=criterion,
        is_classification=is_classification)
    model.add_new_posterior(curvature_scaling, dataloader[0], num_samples=curvature_num_samples)
    return train_metrics
