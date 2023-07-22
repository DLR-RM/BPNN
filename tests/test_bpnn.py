import pytest
from copy import deepcopy
import numpy as np

import torch
from torch import nn, Tensor

from src.curvature.curvatures import BlockDiagonal, Diagonal, KFAC

from src.bpnn.bpnn import BayesianProgressiveNeuralNetwork
from src.bpnn.curvature_scalings import StandardBayes, CurvatureScaling
from src.bpnn.utils import device, compute_curvature

from .conftest import get_test_dataloader

from typing import Any


def list_all_close(list1, list2):
    out = True
    for l1, l2 in zip(list1, list2):
        out = out and torch.allclose(l1, l2)
    return out


def state_dict_close(state_dict1, state_dict2):
    out = True
    out = out and state_dict1.keys() == state_dict2.keys()
    for key in state_dict1.keys():
        out = out and torch.allclose(state_dict1[key], state_dict2[key])
    return out


def curvature_state_close(curvature1, curvature2):
    out = True
    name_to_module2 = dict(curvature2.model.named_modules())
    module1_to_module2 = {module1: name_to_module2[name] for name, module1 in curvature1.model.named_modules()}
    for module1, state1 in curvature1.state.items():
        state2 = curvature2.state[module1_to_module2[module1]]
        out = out and ((isinstance(state1, list) and isinstance(state2, list)) or
                       (isinstance(state1, Tensor) and isinstance(state2, Tensor)))
        if isinstance(state1, list) and isinstance(state2, list):
            out = out and len(state1) == len(state2) == 2
            device = state1[0].device
            state2 = (state2[0].to(device), state2[1].to(device))
            out = out and torch.allclose(state1[0] / state1[0].norm(),
                                         state2[0] / state2[0].norm(), rtol=1e-3, atol=1e-5)
            out = out and torch.allclose(state1[1] / state1[1].norm(),
                                         state2[1] / state2[1].norm(), rtol=1e-3, atol=1e-5)
            out = out and (state1[0].norm() * state1[1].norm()).item() == \
                  pytest.approx((state2[0].norm() * state2[1].norm()).item(), 1e-3)
        else:
            device = state1.device
            out = out and torch.allclose(state1, state2.to(device), rtol=1e-3, atol=1e-5)
    return out


class TestCurvatureScaling(CurvatureScaling):
    def __init__(self, alpha: float, beta: float, temperature_scaling: float):
        self.alpha = alpha
        self.beta = beta
        self.temperature_scaling = temperature_scaling

    def reset(self, *args: Any, **kwargs: Any):
        pass

    def find_optimal_scaling(self, *args: Any, **kwargs: Any):
        return self.alpha, self.beta, self.temperature_scaling


class TestBayesianProgressiveNeuralNetworks:

    def test_init(self, curvature_KFAC):
        bpnn = BayesianProgressiveNeuralNetwork(
            prior=curvature_KFAC,
            backbone=nn.Sequential(
                nn.Conv2d(1, 3, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(1, 3, 3, padding=1),
                nn.ReLU(),
            ),
            last_layer_name='11',
            lateral_connections=['3', '7', '9']
        )
        assert bpnn.base_network == curvature_KFAC.model == bpnn.prior.model
        assert state_dict_close(bpnn.prior.model_state, bpnn.prior.model.state_dict())

    @pytest.fixture()
    def bpnn(self, curvature_KFAC):
        out = BayesianProgressiveNeuralNetwork(
            prior=curvature_KFAC,
            backbone=nn.Sequential(
                nn.Conv2d(1, 3, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(3, 1, 3, padding=1),
                nn.ReLU(),
            ),
            last_layer_name='11',
            lateral_connections=['3', '7', '9']
        )
        return out

    @pytest.fixture()
    def bpnn_multiple_tasks(self, bpnn, dataloader):
        bpnn.add_new_column(is_classification=True, differ_from_previous=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., .5), dataloader)

        bpnn.add_new_column(is_classification=True, differ_from_previous=True)
        bpnn.add_new_posterior(TestCurvatureScaling(2., 1., 1.), dataloader)

        bpnn.add_new_column(is_classification=True, differ_from_previous=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 2., 2.), dataloader)
        return bpnn

    def test_full_state_dict(self, bpnn_multiple_tasks):
        bpnn2 = BayesianProgressiveNeuralNetwork(
            prior=bpnn_multiple_tasks.prior,
            backbone=nn.Sequential(
                nn.Conv2d(1, 3, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(3, 1, 3, padding=1),
                nn.ReLU(),
            ),
            last_layer_name='11',
        )
        bpnn2.load_full_state_dict(bpnn_multiple_tasks.full_state_dict())

        def assert_curvature_close(curvature_column1, curvature_column2):
            name_to_param2 = dict(curvature_column2.model_state)
            for name, param1 in curvature_column1.model_state.items():
                param2 = name_to_param2[name]
                assert torch.allclose(param1, param2.to(param1))

            name_to_module2 = dict(curvature_column2.model.named_modules())
            module1_to_module2 = {module1: name_to_module2[name]
                                  for name, module1 in curvature_column1.model.named_modules()}
            for module1, state1 in curvature_column1.state.items():
                state2 = curvature_column2.state[module1_to_module2[module1]]
                for s1, s2 in zip(state1, state2):
                    assert torch.allclose(s1, s2.to(s1))

                if curvature_column1.inv_state:
                    inv_state1 = curvature_column1.inv_state[module1]
                    inv_state2 = curvature_column2.inv_state[module1_to_module2[module1]]
                    for s1, s2 in zip(inv_state1, inv_state2):
                        assert torch.allclose(s1, s2.to(s1))

        for network1, network2 in zip(bpnn_multiple_tasks.networks, bpnn2.networks):
            for column1, column2 in zip(network1, network2):
                name_to_param2 = dict(column2.named_parameters())
                for name, param1 in column1.named_parameters():
                    param2 = name_to_param2[name]
                    assert torch.allclose(param1, param2.to(param1))

        assert_curvature_close(bpnn_multiple_tasks.prior, bpnn2.prior)
        assert_curvature_close(bpnn_multiple_tasks.prior, bpnn2.prior)

        for curvatures1, curvatures2 in [[bpnn_multiple_tasks.posterior, bpnn2.posterior]]:
            for curvature1, curvature2 in zip(curvatures1, curvatures2):
                for curvature_column1, curvature_column2 in zip(curvature1, curvature2):
                    assert_curvature_close(curvature_column1, curvature_column2)

    def test_add_new_column(self, bpnn):
        assert bpnn.prior
        bpnn.add_new_column(differ_from_previous=False)
        assert len(bpnn.networks) == 1
        assert len(bpnn.networks[0]) == 1
        assert bpnn.networks[0][0] != bpnn.prior.model
        prior_name_to_param = dict(bpnn.prior.model.named_parameters())
        for name, param in bpnn.networks[0][0].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        bpnn.add_new_column(differ_from_previous=False)
        assert len(bpnn.networks) == 2
        assert len(bpnn.networks[1]) == 2
        assert bpnn.networks[1][0] != bpnn.networks[0][0]
        assert bpnn.networks[1][1] != bpnn.prior.model

        prior_name_to_param = dict(bpnn.networks[0][0].named_parameters())
        for name, param in bpnn.networks[1][0].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        prior_name_to_param = dict(bpnn.prior.model.named_parameters())
        for name, param in bpnn.networks[1][1].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        bpnn.add_new_column(differ_from_previous=False)
        assert len(bpnn.networks) == 3
        assert len(bpnn.networks[2]) == 3
        assert bpnn.networks[2][0] != bpnn.networks[0][0]
        assert bpnn.networks[2][1] != bpnn.networks[1][1]
        assert bpnn.networks[2][2] != bpnn.prior.model

        prior_name_to_param = dict(bpnn.networks[0][0].named_parameters())
        for name, param in bpnn.networks[2][0].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        prior_name_to_param = dict(bpnn.networks[1][1].named_parameters())
        for name, param in bpnn.networks[2][1].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        prior_name_to_param = dict(bpnn.prior.model.named_parameters())
        for name, param in bpnn.networks[2][2].named_parameters():
            param_prior = prior_name_to_param[name]
            assert torch.allclose(param, param_prior)

        bpnn.networks = nn.ModuleList([])

    def test_update_scaling(self, bpnn_multiple_tasks):
        def get_scaling(is_float, modules):
            if is_float:
                return torch.rand(1).item()
            else:
                subset_size = np.random.randint(1, len(modules))
                modules_subset = np.random.choice(modules, size=subset_size, replace=False)
                return {module: torch.rand(1).item() for module in modules_subset}

        def assert_scaling_dict(scaling, d, index):
            for module, value in scaling.items():
                if index is None:
                    assert d[module] == value
                else:
                    assert d[module][index] == value

        def assert_scaling_float(scaling, d, update_slice, index):
            for posterior_list in bpnn_multiple_tasks.posterior[update_slice]:
                for posterior in posterior_list:
                    for module in posterior.state.keys():
                        if index is None:
                            assert d[module] == scaling
                        else:
                            assert d[module][index] == scaling

        def assert_scaling(is_float, scaling, d, update_slice, index):
            if is_float:
                assert_scaling_float(scaling, d, update_slice, index)
            else:
                assert_scaling_dict(scaling, d, index)

        update_slice = slice(0, 2)
        modules = list(bpnn_multiple_tasks.modules())
        for temperature_scaling_is_float in [True, False]:
            for alpha_is_float in [True, False]:
                for beta_is_float in [True, False]:
                    temperature_scaling = get_scaling(temperature_scaling_is_float, modules)
                    alpha = get_scaling(alpha_is_float, modules)
                    beta = get_scaling(beta_is_float, modules)
                    bpnn_multiple_tasks.update_scaling(alpha, beta, temperature_scaling, update_slice)
                    assert_scaling(temperature_scaling_is_float, temperature_scaling,
                                   bpnn_multiple_tasks.temperature_scaling, update_slice, None)
                    assert_scaling(alpha_is_float, alpha, bpnn_multiple_tasks.scales,
                                   update_slice, 0)
                    assert_scaling(beta_is_float, beta, bpnn_multiple_tasks.scales,
                                   update_slice, 1)

    def test_update_posterior(self):
        def assert_posterior_close(posterior, likelihood, prior, alpha, beta, weight_decay_layer_names, weight_decay):
            ground_truth = likelihood.add_and_scale(prior,
                                                    [beta, alpha],
                                                    weight_decay_layer_names,
                                                    weight_decay)
            assert curvature_state_close(posterior, ground_truth)

        base_network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        ).to(device)

        curvature = BlockDiagonal(base_network, device=device)
        compute_curvature(base_network, [curvature], get_test_dataloader(20, 10, (10,), 10),
                          invert=False, num_samples=1)

        isotropic = curvature.eye_like(0.1)

        prior = curvature.add_and_scale(isotropic, [1, 1])

        temperature_scaling = 1 / (torch.rand([]) * 1_000)
        weight_decay_layer_names = ['0', '4', '8']
        weight_decay = .1
        alpha = torch.rand([]).item()
        beta = torch.rand([]).item()

        bpnn = BayesianProgressiveNeuralNetwork(prior=prior,
                                                last_layer_name='8',
                                                lateral_connections=['2', '4', '6'],
                                                weight_decay_layer_names=weight_decay_layer_names,
                                                weight_decay=weight_decay,
                                                curvature_device=device)

        bpnn.add_new_column(output_size=15)
        bpnn.add_new_posterior(TestCurvatureScaling(alpha, beta, temperature_scaling),
                               get_test_dataloader(20, 10, (10,), 15))

        curvature00 = BlockDiagonal(bpnn.networks[0][0], device=device)
        compute_curvature(bpnn, [curvature00], get_test_dataloader(20, 10, (10,), 15),
                          invert=False, num_samples=1)

        bpnn.update_posterior(alpha, beta, [curvature00])

        assert_posterior_close(bpnn.posterior[0][0], curvature00, prior, alpha, beta, weight_decay_layer_names,
                               weight_decay)

        bpnn.add_new_column(output_size=10)
        bpnn.add_new_posterior(TestCurvatureScaling(alpha, beta, temperature_scaling),
                               get_test_dataloader(20, 10, (10,), 10))

        curvature10 = BlockDiagonal(bpnn.networks[1][0], device=device)
        curvature11 = BlockDiagonal(bpnn.networks[1][1], device=device)
        compute_curvature(bpnn, [curvature10, curvature11], get_test_dataloader(20, 10, (10,), 10),
                          invert=False, num_samples=1)

        bpnn.update_posterior(alpha, beta, [curvature10, curvature11])

        assert_posterior_close(bpnn.posterior[1][0], curvature10, bpnn.posterior[0][0], alpha, beta,
                               weight_decay_layer_names, weight_decay)
        assert_posterior_close(bpnn.posterior[1][1], curvature11, prior, alpha, beta, weight_decay_layer_names,
                               weight_decay)

        bpnn.add_new_column(output_size=5)
        bpnn.add_new_posterior(TestCurvatureScaling(alpha, beta, temperature_scaling),
                               get_test_dataloader(20, 10, (10,), 5))

        curvature20 = BlockDiagonal(bpnn.networks[2][0], device=device)
        curvature21 = BlockDiagonal(bpnn.networks[2][1], device=device)
        curvature22 = BlockDiagonal(bpnn.networks[2][2], device=device)
        compute_curvature(bpnn, [curvature20, curvature21, curvature22], get_test_dataloader(20, 10, (10,), 5),
                          invert=False, num_samples=1)

        bpnn.update_posterior(alpha, beta, [curvature20, curvature21, curvature22])

        assert_posterior_close(bpnn.posterior[2][0], curvature20, bpnn.posterior[0][0], alpha, beta,
                               weight_decay_layer_names, weight_decay)
        assert_posterior_close(bpnn.posterior[2][1], curvature21, bpnn.posterior[1][1], alpha, beta,
                               weight_decay_layer_names, weight_decay)
        assert_posterior_close(bpnn.posterior[2][2], curvature22, prior, alpha, beta, weight_decay_layer_names,
                               weight_decay)

    def test_sample_and_replace(self, bpnn, dataloader):
        bpnn.add_new_column(differ_from_previous=False)
        model_state = deepcopy(bpnn.networks[0][0].state_dict())

        bpnn.sample_and_replace(slice(None))
        assert state_dict_close(bpnn.networks[0][0].state_dict(), model_state)

        bpnn.add_new_posterior(TestCurvatureScaling(1., 0., 1.), dataloader)

        bpnn.sample_and_replace(slice(-1))
        assert state_dict_close(bpnn.networks[0][0].state_dict(), model_state)

        bpnn.sample_and_replace(slice(None))
        assert not state_dict_close(bpnn.networks[0][0].state_dict(), model_state)

        bpnn.networks[0][0].load_state_dict(model_state)

        bpnn.add_new_column(differ_from_previous=False)
        model_states = [[deepcopy(inter_column.state_dict()) for inter_column in network] for network in bpnn.networks]

        bpnn.sample_and_replace(slice(-2))
        assert state_dict_close(bpnn.networks[0][0].state_dict(), model_states[0][0])
        assert state_dict_close(bpnn.networks[1][0].state_dict(), model_states[1][0])
        assert state_dict_close(bpnn.networks[1][1].state_dict(), model_states[1][1])

        bpnn.sample_and_replace(slice(-1))
        assert not state_dict_close(bpnn.networks[0][0].state_dict(), model_states[0][0])
        assert state_dict_close(bpnn.networks[1][0].state_dict(), model_states[1][0])
        assert state_dict_close(bpnn.networks[1][1].state_dict(), model_states[1][1])

        bpnn.networks[0][0].load_state_dict(model_states[0][0])
        bpnn.add_new_posterior(TestCurvatureScaling(1., 0., 1.), dataloader)

        bpnn.sample_and_replace(slice(-1))
        assert not state_dict_close(bpnn.networks[0][0].state_dict(), model_states[0][0])
        assert state_dict_close(bpnn.networks[1][0].state_dict(), model_states[1][0])
        assert state_dict_close(bpnn.networks[1][1].state_dict(), model_states[1][1])

        bpnn.networks[0][0].load_state_dict(model_states[0][0])

        bpnn.sample_and_replace(slice(None))
        assert not state_dict_close(bpnn.networks[0][0].state_dict(), model_states[0][0])
        assert not state_dict_close(bpnn.networks[1][0].state_dict(), model_states[1][0])
        assert not state_dict_close(bpnn.networks[1][1].state_dict(), model_states[1][1])

    def test_get_quadratic_term(self, bpnn):
        # alpha = beta = 1. as StandardBayes is used

        def _test_one_column():
            assert bpnn.get_quadratic_term(slice(-1)).item() == 0.
            assert bpnn.get_quadratic_term(slice(None)).item() \
                   == bpnn.prior.get_quadratic_term(bpnn.networks[0][0],
                                                    scalings=[],
                                                    weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                    default_weight_decay=bpnn.weight_decay
                                                    ).item()

        bpnn.add_new_column(differ_from_previous=False)
        _test_one_column()

        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        _test_one_column()

        def _test_two_columns():
            assert bpnn.get_quadratic_term(slice(-1)).item() \
                   == bpnn.prior.get_quadratic_term(bpnn.networks[0][0],
                                                    scalings=[],
                                                    weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                    default_weight_decay=bpnn.weight_decay).item()
            assert bpnn.get_quadratic_term(slice(-1, None)).item() \
                   == bpnn.prior.get_quadratic_term(bpnn.networks[1][1],
                                                    scalings=[],
                                                    weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                    default_weight_decay=bpnn.weight_decay).item() \
                   + bpnn.posterior[0][0].get_quadratic_term(bpnn.networks[1][0],
                                                             scalings=[],
                                                             weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                             default_weight_decay=bpnn.weight_decay).item()
            assert bpnn.get_quadratic_term(slice(None)).item() \
                   == pytest.approx(bpnn.prior.get_quadratic_term(bpnn.networks[1][1],
                                                                  scalings=[],
                                                                  weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                  default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[0][0].get_quadratic_term(bpnn.networks[1][0],
                                                                              scalings=[],
                                                                              weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                              default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.prior.get_quadratic_term(bpnn.networks[0][0],
                                                                    scalings=[],
                                                                    weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                    default_weight_decay=bpnn.weight_decay).item())

        bpnn.add_new_column(differ_from_previous=False)
        _test_two_columns()

        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        _test_two_columns()

    @pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
    def test_isotropic_quadratic(self, curvature_type):
        # check against sum(param ** 2 for param in params) for isotropic prior
        base_network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0.)
                m.bias.data.fill_(0.)

        base_network.apply(init_weights)

        curvature = BlockDiagonal(base_network, device=device)
        compute_curvature(base_network, [curvature], get_test_dataloader(20, 10, (10,), 10),
                          invert=False, num_samples=1)

        prior = curvature.eye_like(1.)

        temperature_scaling = 1 / (torch.rand([]) * 1_000)

        bpnn = BayesianProgressiveNeuralNetwork(prior=deepcopy(prior),
                                                last_layer_name='8',
                                                lateral_connections=['2', '4', '6'],
                                                weight_decay_layer_names=[],
                                                weight_decay=0.0,
                                                curvature_device=device)

        bpnn.add_new_column(resample_base_network=True)
        bpnn.add_new_posterior(TestCurvatureScaling(0.5, 0.5, temperature_scaling),
                               get_test_dataloader(200, 100, (10,), 10))
        bpnn.update_posterior(0.5, 0.5, [bpnn.posterior[0][0].eye_like(1.)])
        assert bpnn.get_quadratic_term().item() == pytest.approx(
            sum(param.norm() ** 2 for param in bpnn.networks[0][0].parameters()).item())

        bpnn_not_eye = BayesianProgressiveNeuralNetwork(prior=curvature,
                                                        last_layer_name='8',
                                                        lateral_connections=['2', '4', '6'],
                                                        weight_decay_layer_names=['0', '2', '4', '6', '8'],
                                                        weight_decay=1.0,
                                                        curvature_device=device)
        bpnn_not_eye.add_new_column(resample_base_network=True)
        bpnn_not_eye.add_new_posterior(TestCurvatureScaling(0.5, 0.5, temperature_scaling),
                                       get_test_dataloader(200, 100, (10,), 10))

        assert bpnn_not_eye.get_quadratic_term().item() == pytest.approx(
            sum(param.norm() ** 2 for param in bpnn_not_eye.networks[0][0].parameters()).item())

    def test_kl_divergence(self, bpnn):

        temperature_scaling = 1 / (torch.rand([]) * 1_000)

        def _test_one_column():
            assert bpnn.kl_divergence(slice(-1), temperature_scaling).item() == 0.
            assert bpnn.kl_divergence(slice(None), temperature_scaling).item() \
                   == pytest.approx(bpnn.posterior[0][0].kl_divergence(bpnn.prior,
                                                                       temperature_scaling=temperature_scaling,
                                                                       weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                       default_weight_decay=bpnn.weight_decay).item())

        bpnn.add_new_column(differ_from_previous=True, resample_base_network=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        _test_one_column()

        def _test_two_columns():
            assert bpnn.kl_divergence(slice(-1), temperature_scaling).item() \
                   == pytest.approx(bpnn.posterior[0][0].kl_divergence(bpnn.prior,
                                                                       temperature_scaling=temperature_scaling,
                                                                       weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                       default_weight_decay=bpnn.weight_decay).item())
            assert bpnn.kl_divergence(slice(-1, None), temperature_scaling).item() \
                   == pytest.approx(bpnn.posterior[1][1].kl_divergence(bpnn.prior,
                                                                       temperature_scaling=temperature_scaling,
                                                                       weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                       default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[1][0].kl_divergence(bpnn.posterior[0][0],
                                                                         temperature_scaling=temperature_scaling,
                                                                         weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                         default_weight_decay=bpnn.weight_decay).item())
            assert bpnn.kl_divergence(slice(None), temperature_scaling).item() \
                   == pytest.approx(bpnn.posterior[1][1].kl_divergence(bpnn.prior,
                                                                       temperature_scaling=temperature_scaling,
                                                                       weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                       default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[1][0].kl_divergence(bpnn.posterior[0][0],
                                                                         temperature_scaling=temperature_scaling,
                                                                         weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                         default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[0][0].kl_divergence(bpnn.prior,
                                                                         temperature_scaling=temperature_scaling,
                                                                         weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                         default_weight_decay=bpnn.weight_decay).item())
            assert bpnn.kl_divergence(slice(None)).item() \
                   == pytest.approx(bpnn.posterior[1][1].kl_divergence(bpnn.prior,
                                                                       temperature_scaling=1.,
                                                                       weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                       default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[1][0].kl_divergence(bpnn.posterior[0][0],
                                                                         temperature_scaling=1.,
                                                                         weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                         default_weight_decay=bpnn.weight_decay).item() \
                                    + bpnn.posterior[0][0].kl_divergence(bpnn.prior,
                                                                         temperature_scaling=1.,
                                                                         weight_decay_layer_names=bpnn.weight_decay_layer_names,
                                                                         default_weight_decay=bpnn.weight_decay).item())

        bpnn.add_new_column(differ_from_previous=True, resample_base_network=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        _test_two_columns()

    def test_trace(self, bpnn):

        temperature_scaling = 1 / (torch.rand([]) * 1_000)

        def _test_one_column(curvature_list):
            assert bpnn.trace(curvature_list, temperature_scaling).item() \
                   == pytest.approx(curvature_list[0].trace_of_mm(bpnn.posterior[0][0],
                                                                  temperature_scaling=temperature_scaling).item())

        bpnn.add_new_column(differ_from_previous=True, resample_base_network=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        curvature_list = [
            KFAC(bpnn.networks[0][0], device=device)
        ]
        compute_curvature(bpnn, curvature_list, get_test_dataloader(),
                          invert=False, num_samples=1)
        _test_one_column(curvature_list)

        def _test_two_columns(curvature_list):
            assert bpnn.trace(curvature_list, temperature_scaling).item() \
                   == pytest.approx(curvature_list[0].trace_of_mm(bpnn.posterior[1][0],
                                                                  temperature_scaling=temperature_scaling).item() \
                                    + curvature_list[1].trace_of_mm(bpnn.posterior[1][1],
                                                                    temperature_scaling=temperature_scaling).item())

            assert bpnn.trace(curvature_list).item() \
                   == pytest.approx(curvature_list[0].trace_of_mm(bpnn.posterior[1][0],
                                                                  temperature_scaling=1.).item() \
                                    + curvature_list[1].trace_of_mm(bpnn.posterior[1][1],
                                                                    temperature_scaling=1.).item())

        bpnn.add_new_column(differ_from_previous=True, resample_base_network=True)
        bpnn.add_new_posterior(TestCurvatureScaling(1., 1., 1.), get_test_dataloader())
        curvature_list = [
            KFAC(bpnn.networks[1][0], device=device),
            KFAC(bpnn.networks[1][1], device=device)
        ]
        compute_curvature(bpnn, curvature_list, get_test_dataloader(),
                          invert=False, num_samples=1)
        _test_two_columns(curvature_list)

    def test_temperature_scaling(self, curvature_KFAC):
        temperature_scaling = 1 / (torch.rand([]) * 1_000)
        bpnn = BayesianProgressiveNeuralNetwork(prior=curvature_KFAC,
                                                last_layer_name='11',
                                                lateral_connections=['3', '7', '9'])
        curvature_scaling = StandardBayes()
        bpnn.add_new_column(True, None, True, True)
        bpnn.add_new_posterior(curvature_scaling, get_test_dataloader())

        bpnn_without_ts = deepcopy(bpnn)
        bpnn_without_ts.update_scaling(temperature_scaling=1.)

        assert bpnn.get_quadratic_term() == bpnn_without_ts.get_quadratic_term()

        curvature_list = [
            KFAC(bpnn.networks[0][0], device=device)
        ]
        compute_curvature(bpnn, curvature_list, get_test_dataloader(),
                          invert=False, num_samples=1)
        assert bpnn.trace(curvature_list, temperature_scaling=temperature_scaling).item() / temperature_scaling \
               == pytest.approx(bpnn_without_ts.trace(curvature_list, temperature_scaling=1.).item())

