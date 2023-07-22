import pytest

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.bpnn import compute_curvature
from src.bpnn.utils import device

from src.curvature.lenet5 import lenet5
from src.curvature.curvatures import BlockDiagonal, KFAC


def relative_difference(ground_truth, other, ord=None, dim=None):
    return torch.linalg.norm(ground_truth - other, ord=ord, dim=dim) \
           / (torch.linalg.norm(ground_truth, ord=ord, dim=dim) + 1e-8)


def get_test_dataloader(length=1, batch_size=1, input_shape=(1, 28, 28), num_outputs=10):
    class TestDataset(Dataset):
        def __getitem__(self, item):
            return torch.rand(*input_shape), torch.randint(num_outputs, [])

        def __len__(self):
            return length

    return DataLoader(TestDataset(), batch_size=batch_size)


@pytest.fixture
def test_dataloader():
    return get_test_dataloader(20, 10)

mnist_dataset = MNIST('~/.torch/datasets', False, transform=ToTensor(), download=True)
mnist_dataloader = DataLoader(mnist_dataset, batch_size=250)

def get_small_mnist_dataloader(size=100, batch_size=25):
    small_dataset, _ = random_split(mnist_dataset, [size, len(mnist_dataset) - size])
    out = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)
    return out


@pytest.fixture
def small_mnist_dataloader():
    return get_small_mnist_dataloader()


@pytest.fixture(params=[mnist_dataloader],
                ids=['mnist_dataloader'])
def dataloader(request):
    return request.param


@pytest.fixture
def curvature_KFAC(dataloader):
    net = lenet5(pretrained=True, device=device)
    curvature = KFAC(net)
    compute_curvature(net, [curvature], dataloader, invert=True, num_samples=1)
    return curvature
