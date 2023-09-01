"""Tools to run many model configurations."""
import json
import os
from copy import deepcopy
from typing import Callable, Any, List, Optional, Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.bpnn.bpnn import init_prior, dataset_step_bpnn, \
    BayesianProgressiveNeuralNetwork
from src.bpnn.criterions import CatoniCriterion, MaximumAPosterioriCriterion, \
    McAllesterCriterion, ScaledMaximumAPosterioriCriterion
from src.bpnn.curvature_scalings import CatoniScaling, McAllesterScaling, \
    StandardBayes, CurvatureScaling, ValidationScaling
from src.bpnn.pnn import fit_pnn, evaluate_pnn, ProgressiveNeuralNetwork, \
    dataset_step_pnn, DropoutProgressiveNeuralNetwork, dataset_step_ppnn
from src.bpnn.utils import base_path, seed, device, fit
from src.curvature.curvatures import KFAC, KFOC, Diagonal

model_to_abbreviation = {
    BayesianProgressiveNeuralNetwork: 'BPNN',
    ProgressiveNeuralNetwork: 'PNN',
    DropoutProgressiveNeuralNetwork: 'DPNN'
}


def run_and_save_params_and_output(
        function: Callable[[Any], Any],
        path: Optional[str],
        *args: Any,
        **kwargs: Any):
    """Runs a function and saves the parameters and outputs to a path.

    Args:
        function: The function that should be executed
        path: The path where everything should be saved
        *args: The arguments to the function
        **kwargs: The keyword arguments of the function
    """
    out = {
        'seed': seed,
        'args': args,
        'kwargs': kwargs,
        'metrics': function(*args, **kwargs)
    }
    if path is not None:
        pardir = os.path.abspath(os.path.join(path, os.pardir))
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        with open(path, 'w') as f:
            json.dump(out, f, indent=4, default=str)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_name(name: str = 'unnamed', return_number: bool = False):
    """Searches for the name in the models folder and increments the number
    after the name.

    Args:
        name: A string
        return_number: Whether to return the number or the new name

    Returns:
        The new name with a number at the end or the number
    """
    head, tail = os.path.split(name)
    previous = [int(model_name.split('_')[-1])
                for model_name in os.listdir(os.path.join(base_path, 'models', head))
                if '_'.join(model_name.split('_')[:-1]) == tail]
    num = max(previous) if len(previous) > 0 else 0
    new_name = f'{name}_{num + 1}'
    if return_number:
        return num
    else:
        return new_name


def evaluate_ood(
        model: ProgressiveNeuralNetwork,
        ood_dataloaders: List[Tuple[Optional[int],
        Tuple[DataLoader,
        DataLoader,
        DataLoader]]],
        ood_dims: Iterable[int],
        result_path: str):
    """Evaluates the model on dataloaders and saves the results in an existing
    file.

    Args:
        model: A ProgressiveNeuralNetwork object
        ood_dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        ood_dims: The columns that should be used for each dataloader
        result_path: The path to the file where the results are saved
    """
    if ood_dataloaders and ood_dims:
        with open(result_path, 'r') as f:
            d = json.load(f)
            d['ood'] = evaluate_pnn(model, ood_dataloaders, ood_dims, compute_pac_bounds=False)
        with open(result_path, 'w') as f:
            json.dump(d, f, indent=4, default=str)


def save_scaling(
        model: BayesianProgressiveNeuralNetwork,
        result_path: str):
    """Saves the scaling of the model in an existing file.

    Args:
        model: A BayesianProgressiveNeuralNetwork object
        result_path: The path to the file where the results are saved
    """
    module_to_name = {module: name for name, module in model.named_modules()}
    with open(result_path, 'r') as f:
        d = json.load(f)
        d['scaling'] = {module_to_name[module]: scales
                        for module, scales in model.scales.items()}
        d['temperature scaling'] = {module_to_name[module]: scales
                                    for module, scales in model.temperature_scaling.items()}
    with open(result_path, 'w') as f:
        json.dump(d, f, indent=4, default=str)


def sweep_bpnn(
        prefix: str,
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], nn.Module]],
        network: nn.Module,
        backbone: Optional[nn.Module],
        last_layer_name: str,
        lateral_connections: List[str],
        curvature_types: List[str],
        weight_decays: List[float],
        criterion_types: List[str],
        curvature_scalings: List[str],
        pretrained: bool,
        curvature_device: Optional[torch.device],
        curvature_num_samples: int = 1,
        learning_rate: float = 2e-3,
        num_epochs: int = 100,
        patience: int = 10,
        compute_pac_bounds: bool = True,
        eval_every_task: bool = True,
        confidence: float = 0.8,
        temperature_scalings: Optional[List[float]] = None,
        evaluate_each_temperature: bool = False,
        isotropic_prior: bool = False,
        prior_curvature_scalings: Optional[List[CurvatureScaling]] = None,
        prior_curvature_path: Optional[str] = None,
        prior_curvature_type: Optional[str] = None,
        negative_data_log_likelihood: Optional[float] = None,
        len_data: Optional[int] = None,
        ood_dataloaders: Optional[
            List[Tuple[Optional[int],
            Tuple[DataLoader, DataLoader, DataLoader],
            nn.Module]]] = None,
        ood_dims: Optional[Iterable[int]] = None,
        alphas_betas: Optional[Tuple[List[float], List[float]]] = None,
        validation_scaling_num_samples: Optional[int] = None,
        fixed: Optional[Tuple[float, float, float]] = None,
        shared: Optional[Tuple[bool, bool, bool]] = None,
        curvature_scaling_device: Optional[torch.device] = None,
        checkpoint_path: str = None):
    """Sweeps multiple configurations of Bayesian Progressive Neural Networks.

    All combinations of curvature type, weight decay, temperature scaling, and
    together criterion type and curvature scaling are trained and evaluated.

    The iteration is over criterion type and curvature scaling together. Hence,
    they need to have the same size.

    The results are saved in the results folder for each model individually as a
    JSON file.

    Examples:
        Here, BPNN with the base_network resnet50 are trained on the WRGBD
        dataloaders with two lateral layers and using the fully-connected layer
        as the final layer that is changed in each column dependent on the
        number of classes.
        Alltogether, 24 models are trained all combinations of
            2 curvature types: 'KFAC', 'Diagonal'
            2 weight decays: 1e-8, 1e-6
            2 temperature scalings: 1e-7, 1e-8
            3 combination of criterions to optimize the weights and curvature scalings:
                'MAP' criterion with 'Standard' Bayes curvature scaling
                'MAP' criterion with 'McAllester' curvature scaling
                'Catoni' criterion with 'Catoni' curvature scaling
        >>> network = torchvision.models.resnet50(pretrained=True)
        >>> dataloaders = wrgbd_dataloaders.get_dataloaders(batch_size=4)
        >>> last_layer_name = 'fc'
        >>> lateral_connections = ['layer3.0.downsample.0', 'layer4.0.downsample.0']
        >>> sweep_bpnn(prefix='',
        >>>            dataloaders=dataloaders,
        >>>            network=network,
        >>>            backbone=backbone,
        >>>            last_layer_name=last_layer_name,
        >>>            lateral_connections=lateral_connections,
        >>>            curvature_types=['KFAC', 'Diagonal'],
        >>>            weight_decays=[1e-8, 1e-6],
        >>>            criterion_types=['MAP', 'MAP', 'Catoni'],
        >>>            curvature_scalings=['Standard', 'McAllester', 'Catoni'],
        >>>            pretrained=True,
        >>>            curvature_device=torch.device('cpu'),
        >>>            temperature_scalings=[1e-7, 1e-8],
        >>>            )

    Args:
        prefix: The prefix of the save path of the results and models and the name
        dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        network: A PyTorch model (The base_network)
        backbone: A PyTorch model (The backbone)
        last_layer_name: The name of the last layer
        lateral_connections: The names of the layers that should have
            lateral connections
        curvature_types: A list of curvature names out of 'KFAC', 'KFOC',
            and 'Diagonal'
        weight_decays: A list of weight decays
        criterion_types: A list of criterion names out of 'MAP', 'McALlester',
            and 'Catoni'
        curvature_scalings: A list of curvature scaling names out of 'Standard',
            'McAllester', and 'Catoni'
        pretrained: Whether the model.base_network is already trained
        curvature_device: The device for the curvature
        curvature_num_samples: The number of samples used to compute the curvature
        learning_rate: The learning rate used in the full training
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        compute_pac_bounds: Whether to compute the PAC-Bayes bounds
        eval_every_task: Whether to evaluate the model after each task
        confidence: The confidence used for the PAC Bayes bounds
        temperature_scalings: A list of temperature scalings (scaling of the
            covariance matrices)
        evaluate_each_temperature: Whether to evaluate the model on each
            temperature scaling
        isotropic_prior: Whether to use an isotropic prior
        prior_curvature_scalings: The curvature scaling to scale the prior if
            different from the normal curvature scaling
        prior_curvature_path: The path of a previously computed curvature
        prior_curvature_type: The clas of the prior curvature
        negative_data_log_likelihood: Pre-computed negative log-likelihood
        len_data: Pre-computed length of the dataset
        ood_dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        ood_dims: The columns that should be used for each dataloader
        alphas_betas: A tuple of two lists of alphas and betas for the
            ValidationScaling
        validation_scaling_num_samples: The number of samples used to compute
            the validation scaling
        fixed: A tuple of three floats for fixing alpha, beta, and temperature
        shared: A tuple of three booleans indicating whether the alpha, beta,
            and temperature should be shared or not
        curvature_scaling_device: The device for the curvature scaling
        checkpoint_path: The path where the models should be loaded
    """
    if prior_curvature_scalings is None:
        prior_curvature_scalings = [None]

    name_to_curvature_type = {
        'KFAC': KFAC,
        'KFOC': KFOC,
        'Diagonal': Diagonal
    }

    name_to_criterion_type = {
        'MAP': MaximumAPosterioriCriterion,
        'ScaledMAP': ScaledMaximumAPosterioriCriterion,
        'McAllester': McAllesterCriterion,
        'Catoni': CatoniCriterion
    }

    name_to_curvature_scale = {
        'Standard': StandardBayes,
        'McAllester': McAllesterScaling,
        'Catoni': CatoniScaling,
        'Validation': ValidationScaling,
    }

    if alphas_betas is not None:
        alphas, betas = alphas_betas

    if fixed is None:
        fixed = (None, None, None)

    if shared is None:
        shared = (False, False, False)

    if checkpoint_path:
        pretrained = True
        isotropic_prior = True  # increase speed to compute prior

    for curvature_type_name in curvature_types:
        assert curvature_type_name in name_to_curvature_type.keys()
        curvature_type = name_to_curvature_type[curvature_type_name]
        for weight_decay in weight_decays:
            for criterion_type_name, curvature_scaling_name in zip(criterion_types, curvature_scalings):
                criterion_type = name_to_criterion_type[criterion_type_name]
                if curvature_scaling_name == 'Validation':
                    assert alphas_betas is not None, \
                        'alphas_betas needs to be specified, when using ValidationScaling'
                    assert validation_scaling_num_samples is not None, \
                        'validation_scaling_num_samples needs to be specified, when using ValidationScaling'
                    curvature_scaling = name_to_curvature_scale[curvature_scaling_name](
                        alphas=alphas,
                        betas=betas,
                        temperature_scalings=temperature_scalings,
                        num_samples=validation_scaling_num_samples,
                        dataloader=dataloaders[0][1][1]
                    )
                else:
                    curvature_scaling = name_to_curvature_scale[curvature_scaling_name](
                        confidence=confidence,
                        fixed=fixed,
                        shared=shared,
                    )
                for prior_curvature_scaling in prior_curvature_scalings:

                    name = f'{prefix}BPNN_{curvature_type_name}_' \
                           f'{criterion_type_name}_{weight_decay}_' \
                           f'{curvature_scaling_name}'
                    print(name)

                    base_network = deepcopy(network).to(device)

                    if not pretrained:
                        loss_function = dataloaders[0][2]
                        base_network.requires_grad_(True)
                        fit(
                            base_network, dataloaders[0][1], loss_function, weight_decay,
                            is_classification=type(loss_function) is nn.CrossEntropyLoss,
                            learning_rate=learning_rate, num_epochs=num_epochs, patience=patience)

                    if prior_curvature_type is None:
                        prior_curvature_type = curvature_type
                    if prior_curvature_scaling is None:
                        prior_curvature_scaling = curvature_scaling

                    prior_criterion = dataloaders[0][2]

                    prior_curvature_scaling.reset(
                        dataloader=dataloaders[0][1][1],
                        criterion=prior_criterion,
                        is_classification=type(prior_criterion) == nn.CrossEntropyLoss)

                    prior = init_prior(
                        base_network, prior_curvature_type, dataloaders[0][1][0],
                        weight_decay, prior_curvature_scaling, isotropic_prior,
                        curvature_device, curvature_scaling_device, prior_curvature_path,
                        negative_data_log_likelihood, len_data)

                    model = BayesianProgressiveNeuralNetwork(
                        prior=prior,
                        backbone=backbone,
                        last_layer_name=last_layer_name,
                        lateral_connections=deepcopy(lateral_connections),
                        weight_decay=weight_decay,
                        weight_decay_layer_names=[last_layer_name],
                        curvature_device=curvature_device,
                        curvature_scaling_device=curvature_scaling_device,
                    )

                    if checkpoint_path:
                        number = get_name(name, return_number=True)
                        models_path = os.path.join(base_path, 'models', f'{name}_{number}')
                        id = max(int(el[:-3].split('_')[-1]) for el in os.listdir(models_path) if el.endswith('.pt'))
                        model_path = os.path.join(models_path, f'full_state_dict_{id}.pt')
                        model.load_full_state_dict(torch.load(model_path))

                    criterion = criterion_type(model, confidence=confidence)

                    result_path = os.path.join(base_path, 'results', name + '.json')
                    run_and_save_params_and_output(
                        function=fit_pnn,
                        path=result_path,
                        model=model,
                        dataloaders=dataloaders,
                        name=get_name(name),
                        eval_every_task=eval_every_task,
                        dataset_step=dataset_step_bpnn,
                        curvature_type=curvature_type,
                        criterion=criterion,
                        weight_decay=weight_decay,
                        curvature_scaling=curvature_scaling,
                        curvature_num_samples=curvature_num_samples,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs,
                        patience=patience,
                        compute_pac_bounds=compute_pac_bounds,
                        confidence=confidence
                    )
                    save_scaling(model, result_path)
                    evaluate_ood(model, ood_dataloaders, ood_dims, result_path)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if evaluate_each_temperature:
                        for temperature_scaling in temperature_scalings:
                            model.update_scaling(
                                temperature_scaling=temperature_scaling,
                                update_slice=slice(None))
                            name = f'{prefix}BPNN_{curvature_type_name}_' \
                                   f'{criterion_type_name}_{weight_decay}_' \
                                   f'{curvature_scaling_name}_{temperature_scaling}'
                            print(name)

                            result_path = os.path.join(base_path, 'results', name + '.json')
                            d = {}
                            is_classification = [type(loss_function) is type(nn.CrossEntropyLoss())
                                                 for _, _, loss_function in dataloaders[1:]]
                            d['metrics'] = [{
                                                'test': evaluate_pnn(
                                                    model, dataloaders[1:],
                                                    range(0, len(dataloaders) - 1),
                                                    compute_pac_bounds=compute_pac_bounds,
                                                    confidence=confidence,
                                                    is_classification=is_classification)
                                            }]
                            with open(result_path, 'w') as f:
                                json.dump(d, f, indent=4, default=str)
                            evaluate_ood(model, ood_dataloaders, ood_dims, result_path)


def sweep_dpnn(
        prefix: str,
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], nn.Module]],
        network: nn.Module,
        backbone: Optional[nn.Module],
        last_layer_name: str,
        lateral_connections: List[str],
        pretrained: bool,
        weight_decays: List[float],
        dropout_probabilities: List[float],
        dropout_positions: List[str],
        learning_rate: float = 2e-3,
        num_epochs: int = 100,
        patience: int = 10,
        eval_every_task: bool = True,
        ood_dataloaders: Optional[
            List[Tuple[Optional[int],
            Tuple[DataLoader, DataLoader, DataLoader],
            nn.Module]]] = None,
        ood_dims: Optional[Iterable[int]] = None):
    """Sweeps multiple configurations of Dropout Progressive Neural Networks.

    The results are saved in the results folder for each model individually as a
    JSON file.

    Examples:
        Here, PNN with MC Dropout with the base_network resnet50 are trained on
        the WRGBD dataloaders with two lateral layers and using the
        fully-connected layer as the final layer that is changed in each column
        dependent on the number of classes.
        Alltogether, 6 models are trained all combinations of
            2 weight decays: 1e-4, 1e-2
            3 dropout probabilities: .01, .1, .3
        >>> network = torchvision.models.resnet50(pretrained=True)
        >>> dataloaders = wrgbd_dataloaders.get_dataloaders(batch_size=4)
        >>> last_layer_name = 'fc'
        >>> lateral_connections = ['layer3.0.downsample.0', 'layer4.0.downsample.0']
        >>> sweep_pnn(prefix='',
        >>>           dataloaders=dataloaders,
        >>>           network=network,
        >>>           last_layer_name=last_layer_name,
        >>>           lateral_connections=lateral_connections,
        >>>           pretrained=True,
        >>>           weight_decays=[1e-4, 1e-2],
        >>>           dropout_probabilities=[.01, .1, .3],
        >>>           dropout_positions=['layer4.1.conv1', 'layer4.2.conv3'],
        >>>           )

    Args:
        prefix: The prefix of the save path of the results and models and the name
        dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        network: A PyTorch model (The base_network)
        backbone: A PyTorch model (The backbone)
        last_layer_name: The name of the last layer
        lateral_connections: The names of the layers that should have
            lateral connections
        pretrained: Whether the model.base_network is already trained
        weight_decays: A list of weight decays
        dropout_probabilities: A list of probabilities of the dropout layers
        dropout_positions: The layer names where a dropout layer should be appended
        learning_rate: The learning rate used in the full training
        num_epochs: The number of epochs used in the full training
        patience: The patience used in the full training
        eval_every_task: Whether to evaluate the model after each task
        ood_dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        ood_dims: The columns that should be used for each dataloader
    """
    for weight_decay in weight_decays:
        for dropout_probability in dropout_probabilities:
            name = f'{prefix}DPNN_{dropout_probability}_{weight_decay}'
            print(name)

            base_network = deepcopy(network).to(device)

            if not pretrained:
                loss_function = dataloaders[0][2]
                base_network.requires_grad_(True)
                fit(
                    base_network, dataloaders[0][1], loss_function, weight_decay,
                    is_classification=type(loss_function) is nn.CrossEntropyLoss,
                    learning_rate=learning_rate, num_epochs=num_epochs, patience=patience)

            model = DropoutProgressiveNeuralNetwork(
                base_network=base_network,
                backbone=backbone,
                last_layer_name=last_layer_name,
                lateral_connections=deepcopy(lateral_connections),
                dropout_probability=dropout_probability,
                dropout_positions=dropout_positions)
            result_path = os.path.join(base_path, 'results', name + '.json')
            run_and_save_params_and_output(
                function=fit_pnn,
                path=result_path,
                model=model,
                dataloaders=dataloaders,
                name=get_name(name),
                eval_every_task=eval_every_task,
                dataset_step=dataset_step_ppnn,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                patience=patience,
                pretrained=pretrained,
                ood_dataloaders=ood_dataloaders,
                ood_dims=ood_dims,
                compute_pac_bounds=False)
            evaluate_ood(model, ood_dataloaders, ood_dims, result_path)


def sweep_pnn(
        prefix: str,
        dataloaders: List[Tuple[Optional[int], Tuple[DataLoader, DataLoader, DataLoader], nn.Module]],
        network: nn.Module,
        backbone: Optional[nn.Module],
        last_layer_name: str,
        lateral_connections: List[str],
        pretrained: bool,
        weight_decays: List[float],
        learning_rate: float = 2e-3,
        num_epochs: int = 100,
        patience: int = 10,
        eval_every_task: bool = True,
        ood_dataloaders: Optional[
            List[Tuple[Optional[int],
            Tuple[DataLoader, DataLoader, DataLoader],
            nn.Module]]] = None,
        ood_dims: Optional[Iterable[int]] = None):
    """Sweeps multiple configurations of Progressive Neural Networks.

    The results are saved in the results folder for each model individually as a
    JSON file.

    Examples:
        Here, PNN with the base_network resnet50 are trained on the WRGBD
        dataloaders with two lateral layers and using the fully-connected layer
        as the final layer that is changed in each column dependent on the
        number of classes.
        Alltogether, 3 models, one for each weight decay 1e-4, 1e-2, 0 are
        trained.
        >>> network = torchvision.models.resnet50(pretrained=True)
        >>> dataloaders = wrgbd_dataloaders.get_dataloaders(batch_size=4)
        >>> last_layer_name = 'fc'
        >>> lateral_connections = ['layer3.0.downsample.0', 'layer4.0.downsample.0']
        >>> sweep_pnn(prefix='',
        >>>           dataloaders=dataloaders,
        >>>           network=network,
        >>>           last_layer_name=last_layer_name,
        >>>           lateral_connections=lateral_connections,
        >>>           pretrained=True,
        >>>           weight_decays=[1e-4, 1e-2, 0.],
        >>>           )

    Args:
        prefix: The prefix of the save path of the results and models and the name
        dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        network: A PyTorch model (The base_network)
        backbone: A PyTorch model (The backbone)
        last_layer_name: The name of the last layer
        lateral_connections: The names of the layers that should have
            lateral connections
        pretrained: Whether the model.base_network is already trained
        weight_decays: A list of weight decays
        learning_rate: The learning rate used in the full training
        num_epochs: The number of epochs used in the full training
        patience: The patience used in the full training
        eval_every_task: Whether to evaluate the model after each task
        ood_dataloaders: A tuple of output dimensions and train-, val-, and
            test-dataloaders
        ood_dims: The columns that should be used for each dataloader
    """
    for weight_decay in weight_decays:
        name = f'{prefix}PNN_{weight_decay}'
        print(name)

        base_network = deepcopy(network).to(device)

        if not pretrained:
            loss_function = dataloaders[0][2]
            base_network.requires_grad_(True)
            fit(
                base_network, dataloaders[0][1], loss_function, weight_decay,
                is_classification=type(loss_function) is nn.CrossEntropyLoss,
                learning_rate=learning_rate, num_epochs=num_epochs, patience=patience)

        model = ProgressiveNeuralNetwork(
            base_network=base_network,
            backbone=backbone,
            last_layer_name=last_layer_name,
            lateral_connections=deepcopy(lateral_connections))
        result_path = os.path.join(base_path, 'results', name + '.json')
        run_and_save_params_and_output(
            function=fit_pnn,
            path=result_path,
            model=model,
            dataloaders=dataloaders,
            name=get_name(name),
            eval_every_task=eval_every_task,
            dataset_step=dataset_step_pnn,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            pretrained=pretrained,
            ood_dataloaders=ood_dataloaders,
            ood_dims=ood_dims,
            compute_pac_bounds=False)
        evaluate_ood(model, ood_dataloaders, ood_dims, result_path)
