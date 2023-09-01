from math import sqrt

import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal, kl_divergence
from torch.distributions.constraints import positive_definite
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.bpnn import compute_curvature
from src.bpnn.utils import torch_data_path
from src.curvature.curvatures import BlockDiagonal, Diagonal, KFAC, KFOC
from src.curvature.utils import check_and_make_pd
from .conftest import get_small_mnist_dataloader, relative_difference, get_test_dataloader


@pytest.fixture()
def device():
    return torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@pytest.fixture()
def batch_1_dataloader():
    dataset = MNIST(torch_data_path, train=True, transform=ToTensor())
    return DataLoader(dataset, batch_size=1, shuffle=True)


def vectorize(module):
    weight = module.weight.view(module.weight.shape[0], -1)
    if module.bias is not None:
        weight = torch.cat([weight, module.bias[:, None]], dim=1)
    weight = weight.view(-1)
    return weight


eps = 1e-3


# test Block diagonal -> set gradient to 1 for one position, to 0 else and weight the same
# check if 1 comes out

def small_model():
    model = nn.Sequential(
        nn.Conv2d(1, 2, 11),
        nn.ReLU(),
        nn.Conv2d(2, 3, 7),
        nn.ReLU(),
        nn.Conv2d(3, 2, 5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    )
    return model


def get_fisher(state, dtype=torch.float):
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


@pytest.mark.skip(reason="sampling takes very long")
@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, KFOC, BlockDiagonal])
def test_sample_and_replace(curvature_type):
    temperature_scaling = 1.  # torch.rand([]).item() * 1_000

    linear = nn.Linear(2, 4)
    curv = curvature_type(linear)
    nn.Softmax()(linear(torch.rand(20, 2))).log().sum().backward()
    curv.update(20)
    curv.invert()

    mean = vectorize(linear)

    samples = []
    num_samples = 10_000_000
    for _ in range(num_samples):
        curv.sample_and_replace(temperature_scaling)
        samples.append(vectorize(linear))
    samples = torch.stack(samples)

    mean_empirical = torch.mean(samples, dim=0)
    assert torch.allclose(mean_empirical, mean, rtol=1e-1, atol=1e-1)

    covariance = (samples - mean_empirical[None, ...]).T @ (samples - mean_empirical[None, ...]) / samples.shape[0]
    fisher = get_fisher(curv.inv_state[linear])
    fisher = fisher @ fisher.T
    fisher *= temperature_scaling
    assert torch.allclose(covariance, fisher, rtol=2e-1, atol=1e-1)


def test_update_block_diagonal(device, test_dataloader):
    # smaller model as BlockDiagonal with lenet can not be computed for the first Linear layer
    model = nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        nn.Flatten(),
        torch.nn.Linear(16 * 5 * 5, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10)
    )
    block_diagonal = BlockDiagonal(model)
    block_diagonal_batch_size_1 = BlockDiagonal(model)

    old_batch_size_1_values = {}

    model.to(device)
    for features, _ in test_dataloader:
        features = features.to(device)
        logits = model(features)

        sampled_labels = torch.ones(logits.shape[0], dtype=torch.long)
        loss = nn.CrossEntropyLoss()(logits, sampled_labels)
        model.zero_grad()
        loss.backward()
        block_diagonal.update(batch_size=features.size(0))
        for feature in features:
            logit = model(feature[None, ...])

            sampled_label = torch.ones(1, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(logit, sampled_label)
            model.zero_grad()
            loss.backward()
            block_diagonal_batch_size_1.update(batch_size=features.size(0))

            for module in model.modules():
                if module._get_name() in ['Linear', 'Conv2d']:
                    state = block_diagonal_batch_size_1.state[module].clone()
                    if module in old_batch_size_1_values.keys():
                        state_old = old_batch_size_1_values[module].clone()
                        old_batch_size_1_values[module] = state.clone()
                        state -= state_old
                    else:
                        old_batch_size_1_values[module] = state.clone()

                    grad = module.weight.grad.view(module.weight.grad.shape[0], -1)
                    if module.bias is not None:
                        grad = torch.cat([grad, module.bias.grad[:, None]], dim=1)
                    grad = grad.view(-1)
                    assert torch.allclose(state, grad.outer(grad))

    block_diagonal.scale(num_batches=len(test_dataloader), make_pd=False)
    block_diagonal_batch_size_1.scale(num_batches=len(test_dataloader.dataset), make_pd=False)

    for module in block_diagonal.state.keys():
        block_diagonal_state = block_diagonal.state[module]
        block_diagonal_batch_size_1_state = block_diagonal_batch_size_1.state[module]

        assert torch.allclose(block_diagonal_batch_size_1_state, block_diagonal_state, atol=1e-3)


@pytest.mark.parametrize('dataloader',
                         [get_small_mnist_dataloader(1, 1), get_small_mnist_dataloader(100, 25)],
                         ids=['batch_size1', 'batch_size25'])
def test_update(dataloader):
    device = torch.device('cpu')
    model = small_model()

    block_diagonal = BlockDiagonal(model)
    diagonal = Diagonal(model)
    kfac = KFAC(model)
    kfoc = KFOC(model)

    compute_curvature(model, [block_diagonal, diagonal, kfac, kfoc], dataloader,
                      num_samples=1, make_positive_definite=False, device=device)

    for module in block_diagonal.state.keys():
        block_diagonal_state = block_diagonal.state[module]
        diagonal_state = diagonal.state[module].view(-1)
        kfac_state = kfac.state[module]
        kfoc_state = kfoc.state[module]

        assert torch.allclose(diagonal_state, torch.diag(block_diagonal_state), atol=1e-3)
        if module._get_name() == 'Linear' and module.in_features == 16 * 5 * 5:
            continue
        if dataloader.batch_size == 1 and type(module).__name__ == 'Linear':
            assert torch.allclose(get_fisher(kfac_state), block_diagonal_state, atol=1e-3)
            assert torch.allclose(get_fisher(kfoc_state), block_diagonal_state, atol=1e-3)
        else:
            rdiff_kfac = relative_difference(block_diagonal_state, get_fisher(kfac_state)).item()
            rdiff_kfoc = relative_difference(block_diagonal_state, get_fisher(kfoc_state)).item()

            print(f'KFAC: {rdiff_kfac}')
            print(f'KFOC: {rdiff_kfoc}')

            assert rdiff_kfoc < 1. + eps, module
            assert rdiff_kfoc < rdiff_kfac + eps, module


def check_and_make_pd_local(state, eps=1e-6, dtype=torch.float):
    assert isinstance(state, torch.Tensor) or isinstance(state, list)
    if isinstance(state, torch.Tensor):
        assert state.ndim in [1, 2]
        state = state.to(dtype)
        if state.ndim == 1:
            state[state < eps] = eps
            return state
        elif state.ndim == 2:
            if state.shape[0] == state.shape[1]:
                state = check_and_make_pd(state, eps)
                assert positive_definite.check(state)
                return state
            else:
                state[state < eps] = eps
                return state
    elif isinstance(state, list):
        return [check_and_make_pd_local(state_i, eps, dtype) for state_i in state]


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_invert(curvature_type, dataloader):
    # eigenvalues of matrix * inverse == 1.

    model = small_model()

    curvature = curvature_type(nn.ModuleList([model[4], model[9]]))
    compute_curvature(model, [curvature], dataloader, invert=False)

    for module, state in curvature.state.items():
        curvature.state[module] = check_and_make_pd_local(state, dtype=torch.float)
    curvature.invert()

    for module in curvature.state.keys():
        state = curvature.state[module]
        inv_state_chol = curvature.inv_state[module]
        fisher = get_fisher(state, dtype=torch.double)
        inv_fisher_chol = get_fisher(inv_state_chol, dtype=torch.double)
        ground_truth_inverse = check_and_make_pd(torch.inverse(fisher))
        r_diff_inv = relative_difference(ground_truth_inverse, inv_fisher_chol @ inv_fisher_chol.T).item()
        r_diff_eye = relative_difference(torch.eye(*fisher.shape, device=fisher.device, dtype=fisher.dtype),
                                         fisher @ inv_fisher_chol @ inv_fisher_chol.T).item()
        if r_diff_inv < 5e-4:
            print(r_diff_inv, r_diff_eye)
        assert r_diff_inv < 5e-4 or r_diff_eye < .6


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_get_quadratic_term(curvature_type, dataloader):
    # compute KL divergence with same covariance matrices
    model1 = small_model()
    model2 = small_model()
    model3 = small_model()

    weight_decay_layer_names = [name for name, _ in model1.named_modules()][1:None:3]
    default_weight_decay = torch.rand([]).item()

    curv2 = curvature_type(model2)
    compute_curvature(model2, [curv2], dataloader, invert=False, make_positive_definite=True)

    for module, state in curv2.state.items():
        curv2.state[module] = check_and_make_pd_local(state)

    scalings = [{module: torch.rand([]).item()
                 for module in model1.modules()
                 if module._get_name() in ['Conv2d', 'Linear']}
                for _ in range(5)] + [10.]
    weight_decays = [{module: torch.rand([]).item()
                      for module in model1.modules()
                      if module._get_name() in ['Conv2d', 'Linear']}
                     for _ in range(5)] + [5.]
    quadratic_term = curv2.get_quadratic_term(model1, model3.state_dict(),
                                              scalings=scalings,
                                              weight_decays=weight_decays,
                                              weight_decay_layer_names=weight_decay_layer_names,
                                              default_weight_decay=default_weight_decay)

    module2_to_name = {module: name for name, module in model2.named_modules()}

    name_to_module1 = dict(model1.named_modules())
    name_to_module3 = dict(model3.named_modules())

    precision_matrices = []
    mean1 = []
    mean2 = []

    for module, state in curv2.state.items():
        module1 = name_to_module1[module2_to_name[module]]
        scale = 1.
        for scaling in scalings:
            scale *= scaling[module1] if isinstance(scaling, dict) else scaling
        weight_decay = 1.
        for wd in weight_decays:
            weight_decay *= wd[module1] if isinstance(wd, dict) else wd

        if module2_to_name[module] in weight_decay_layer_names:
            scale = 0.
            weight_decay = default_weight_decay
        fisher = get_fisher(state)
        fisher *= scale
        fisher += weight_decay * torch.eye(*fisher.shape, dtype=fisher.dtype, device=fisher.device)

        fisher = check_and_make_pd(fisher)
        precision_matrices.append(fisher)

        mean1.append(vectorize(module1))
        if module2_to_name[module] in weight_decay_layer_names:
            mean2.append(torch.zeros_like(mean1[-1]))
        else:
            mean2.append(vectorize(name_to_module3[module2_to_name[module]]))

    mean1 = torch.cat(mean1).to(torch.device('cpu'))
    mean2 = torch.cat(mean2).to(torch.device('cpu'))

    precision_matrix = torch.block_diag(*precision_matrices).to(torch.device('cpu'))

    normal1 = MultivariateNormal(mean1, precision_matrix=precision_matrix)
    normal2 = MultivariateNormal(mean2, precision_matrix=precision_matrix)

    kl = kl_divergence(normal1, normal2)

    assert .5 * quadratic_term.item() == pytest.approx(kl.item(), rel=1e-2)


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_kl_divergence(curvature_type):
    model1 = small_model()
    model2 = small_model()

    curv1 = curvature_type(model1)
    curv2 = curvature_type(model2)

    compute_curvature(model1, [curv1], get_test_dataloader(20, 10))
    compute_curvature(model2, [curv2], get_test_dataloader(20, 10))

    for curv in [curv1, curv2]:
        for module, state in curv.state.items():
            new_state = check_and_make_pd_local(state)
            curv.state[module] = new_state
        curv.invert()

    weight_decay_layer_names = ['2', '9']
    default_weight_decay = torch.rand([]).item()
    temperature_scaling = {module: 1 / (torch.rand([]).item() * 1000)
                           for module in curv1.state.keys()}

    computed = curv1.kl_divergence(curv2, temperature_scaling, weight_decay_layer_names, default_weight_decay).item()

    expected = 0.

    name_to_module2 = dict(model2.named_modules())
    for name, module1 in model1.named_modules():
        if module1 in curv1.inv_state:
            inv_state1 = curv1.inv_state[module1]
            module2 = name_to_module2[name]

            mean1 = vectorize(module1)
            scale_tril1 = get_fisher(inv_state1) * sqrt(temperature_scaling[module1])

            if name in weight_decay_layer_names:
                mean2 = torch.zeros_like(mean1)
                scale_tril2 = torch.eye(*scale_tril1.shape).to(scale_tril1) / sqrt(
                    default_weight_decay)
            else:
                mean2 = vectorize(module2)
                inv_state2 = curv2.inv_state[module2]
                scale_tril2 = get_fisher(inv_state2)

            normal1 = MultivariateNormal(mean1,
                                         scale_tril=scale_tril1)
            normal2 = MultivariateNormal(mean2,
                                         scale_tril=scale_tril2)
            expected += kl_divergence(normal1, normal2).item()

    assert computed == pytest.approx(expected, rel=1e-3)


def to_dtype(curv, dtype):
    for module, state in curv.state.items():
        if isinstance(state, list):
            curv.state[module] = [s.to(dtype) for s in state]
            if module in curv.inv_state.keys():
                curv.inv_state[module] = [s.to(dtype) for s in curv.inv_state[module]]
        else:
            curv.state[module] = state.to(dtype)
            if module in curv.inv_state.keys():
                curv.inv_state[module] = curv.inv_state[module].to(dtype)


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_trace_of_mm(curvature_type, dataloader):
    model1 = small_model()
    curv1 = curvature_type(model1)
    compute_curvature(model1, [curv1], dataloader, invert=False, make_positive_definite=True)

    model2 = small_model()
    curv2 = curvature_type(model2)
    compute_curvature(model2, [curv2], dataloader, invert=True, make_positive_definite=True)

    temperature_scaling = {module: 1 / (torch.rand([]).item() * 1000)
                           for module in curv1.state.keys()}

    name_to_module2 = dict(model2.named_modules())

    module1_to_module2 = {module: name_to_module2[name] for name, module in model1.named_modules()}

    to_dtype(curv1, torch.double)
    to_dtype(curv2, torch.double)

    trace = 0.
    for module1, state1 in curv1.state.items():
        module2 = module1_to_module2[module1]
        state2 = curv2.state[module2]

        fisher1 = get_fisher(state1, dtype=torch.double)
        fisher2 = get_fisher(state2, dtype=torch.double)
        mm = temperature_scaling[module1] * fisher1 @ fisher2.inverse()

        trace += mm.trace().item()

    computed = curv1.trace_of_mm(curv2, temperature_scaling).item()

    assert computed == pytest.approx(trace, rel=1e-2)


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_eye_like(curvature_type, test_dataloader):
    # test inverse same as state, state close to eye
    model = small_model()
    curv = curvature_type(model)
    compute_curvature(model, [curv], test_dataloader)
    eye = curv.eye_like()

    for module, state in eye.state.items():
        fisher = get_fisher(state)
        assert torch.allclose(fisher, torch.eye(*fisher.shape, dtype=fisher.dtype, device=fisher.device))

        inv_fisher = get_fisher(eye.inv_state[module])
        assert torch.allclose(inv_fisher,
                              torch.eye(*inv_fisher.shape, dtype=inv_fisher.dtype, device=inv_fisher.device))

    weight_decay = torch.rand([]).item()
    eye2 = curv.eye_like(weight_decay=weight_decay)
    quadratic_term = eye2.get_quadratic_term(model)

    quadratic_term_gt = weight_decay * sum(param.pow(2).sum() for param in model.parameters())

    assert quadratic_term.item() == pytest.approx(quadratic_term_gt.item(), rel=1e-3)


@pytest.mark.parametrize('curvature_type', [Diagonal, KFAC, BlockDiagonal])
def test_add_and_scale(curvature_type, dataloader):
    model1 = small_model()
    curv1 = curvature_type(model1)
    compute_curvature(model1, [curv1], dataloader, invert=False, make_positive_definite=True)

    model2 = small_model()
    curv2 = curvature_type(model2)
    compute_curvature(model2, [curv2], dataloader, invert=True, make_positive_definite=True)

    weight_decay_layer_names = ['0', '4', '9']
    weight_decay = 1 / (torch.rand([]).item() * 1000)

    scaling = [{module: torch.rand([]).item() for module in curv1.state.keys()} for _ in range(2)]
    curv3 = curv1.add_and_scale(curv2, scaling, weight_decay=weight_decay,
                                weight_decay_layer_names=weight_decay_layer_names)

    name_to_module_2 = dict(model2.named_modules())

    for name, module in model1.named_modules():
        if module in curv1.state:
            if curvature_type in [Diagonal, BlockDiagonal]:
                fisher1 = get_fisher(curv1.state[module])
                fisher2 = get_fisher(curv2.state[name_to_module_2[name]])
                fisher3 = get_fisher(curv3.state[module])

                if name in weight_decay_layer_names:
                    fisher2 = weight_decay * torch.eye(*fisher2.shape, dtype=fisher2.dtype, device=fisher2.device)

                assert torch.allclose(fisher3, scaling[0][module] * fisher1 + scaling[1][module] * fisher2, rtol=1e-3)
            else:
                if name in weight_decay_layer_names:
                    state2 = curv2._eye_state(curv2.state[name_to_module_2[name]], weight_decay)
                else:
                    state2 = curv2.state[name_to_module_2[name]]
                actual = curv3.state[module]
                expected = KFAC._sum_state([curv1.state[module], state2], [scaling[0][module], scaling[1][module]])
                for a, e in zip(actual, expected):
                    assert torch.allclose(a, e, rtol=1e-3)
