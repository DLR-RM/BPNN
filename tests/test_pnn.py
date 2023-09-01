from copy import deepcopy
from random import randint

import pytest
import torch
from torch import nn

from src.bpnn.pnn import ProgressiveNeuralNetwork, \
    ProbabilisticProgressiveNeuralNetwork, DropoutProgressiveNeuralNetwork


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 20)
        self.linear4 = nn.Linear(20, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = self.linear1(x)
        b = self.linear3(self.relu(self.linear2(self.relu(a))))
        return self.linear4(a + b)


@pytest.fixture
def network():
    return SimpleNet()


def pnn_from_network(network):
    lateral_connections = ['linear2', 'linear3']
    return ProgressiveNeuralNetwork(network, None, 'linear4', lateral_connections)


def simple_net_forward(networks, x):
    a001 = networks[0][0].relu(networks[0][0].linear1(x))
    a002 = networks[0][0].relu(networks[0][0].linear2[0](a001))
    h111 = networks[1][1].linear1(x)
    a112 = networks[1][1].relu((networks[1][0]._modules['linear2']['0'](a001) +
                                networks[1][1].linear2[0](networks[1][1].relu(h111))) / 2)
    return networks[1][1].linear4((networks[1][1].linear3[0](a112)
                                   + networks[1][0]._modules['linear3']['0'](a002)) / 2
                                  + h111)


def weighted_module_same(module1, module2):
    return torch.all(module1.weight == module2.weight).item() \
        and torch.all(module1.bias == module2.bias).item()


def weighted_module_close(module1, module2):
    return torch.allclose(module1.weight, module2.weight, rtol=1e-3) \
        and torch.allclose(module1.bias, module2.bias, rtol=1e-3)


def assert_networks_same(network1, network2):
    name_to_module1 = dict(network1.named_modules())
    name_to_module2 = dict(network2.named_modules())

    assert name_to_module1.keys() == name_to_module2.keys()

    for name, module1 in name_to_module1.items():
        module2 = name_to_module2[name]
        assert type(module1) == type(module2)

        if hasattr(module1, 'weight'):
            assert torch.all(module1.weight == module2.weight).item()
            if hasattr(module1, 'bias'):
                assert torch.all(module1.weight == module2.weight).item()


class TestProgressiveNeuralNetworks:
    @pytest.mark.parametrize(['differ_from_previous', 'resample_base_network'],
                             [[True, True], [True, False], [False, False], [False, True]])
    def test_add_new_column_and_forward(self, differ_from_previous, resample_base_network, network):
        pnn = pnn_from_network(network)
        pnn.add_new_column(True, differ_from_previous=differ_from_previous, resample_base_network=resample_base_network)
        pnn.add_new_column(False, 2, differ_from_previous=differ_from_previous,
                           resample_base_network=resample_base_network)
        print(pnn.is_classification)
        assert pnn.is_classification[0] and not pnn.is_classification[1]
        # test resample base network and differ from previous
        for column in pnn.networks:
            net = column[-1]
            net_name_to_module = dict(net.named_modules())
            for name, base_module in pnn.base_network.named_modules():
                if hasattr(base_module, 'weight') and name != 'linear4':
                    assert weighted_module_same(base_module, net_name_to_module[name]) != (differ_from_previous
                                                                                           or resample_base_network)
                    assert weighted_module_close(base_module, net_name_to_module[name]) != resample_base_network

        for lateral_connection in ['linear2', 'linear3']:
            assert weighted_module_same(pnn.networks[0][0]._modules[lateral_connection][0],
                                        pnn.networks[1][0]._modules[lateral_connection]['0']) != differ_from_previous
            assert weighted_module_close(pnn.networks[0][0]._modules[lateral_connection][0],
                                         pnn.networks[1][0]._modules[lateral_connection]['0'])

        # test forward
        x = torch.rand(5, 10)
        assert torch.allclose(simple_net_forward(pnn.networks, x), pnn(x)[1])
        pnn.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
        assert torch.allclose(simple_net_forward(pnn.networks, x), pnn(x)[1])

    def test_full_state_dict(self, network):
        network_copy = deepcopy(network)
        pnn = pnn_from_network(network)

        pnn.add_new_column(True, differ_from_previous=True, resample_base_network=True)
        pnn.add_new_column(False, 2, differ_from_previous=True, resample_base_network=True)
        pnn.add_new_column(True, 3, differ_from_previous=True, resample_base_network=True)

        full_state_dict = pnn.full_state_dict()

        pnn_copy = ProgressiveNeuralNetwork(network_copy, None, None, None)
        pnn_copy.load_full_state_dict(full_state_dict)

        for _ in range(10):
            x = torch.rand(5, 10)
            for out, out_copy in zip(pnn(x), pnn_copy(x)):
                assert torch.allclose(out, out_copy)

    def test_train(self, network):
        pnn = pnn_from_network(network)
        for _ in range(5):
            pnn.add_new_column(differ_from_previous=True, resample_base_network=True)
            assert pnn.training and pnn.networks[-1].training and not any(net.training for net in pnn.networks[:-1])

            pnn.eval()
            assert not pnn.training and not pnn.networks[-1].training and not any(
                net.training for net in pnn.networks[:-1])

            pnn.train()
            assert pnn.training and pnn.networks[-1].training and not any(net.training for net in pnn.networks[:-1])

    def test_backbone(self):
        backbone = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))
        pnn = ProgressiveNeuralNetwork(nn.Identity(), backbone, '0', [])
        pnn.add_new_column()
        pnn.add_new_column()
        assert pnn.backbone == backbone
        for _ in range(10):
            x = torch.rand(5, 10)
            out_gt = backbone(x)
            for out in pnn(x):
                assert torch.allclose(out, out_gt)
        pnn.train()
        assert not pnn.backbone.training
        pnn.eval()
        assert not pnn.backbone.training

        for module in pnn.modules():
            assert module not in pnn.backbone.modules()


class TestProbabilisticProgressiveNeuralNetwork:

    def ppnn_and_record(self, network, train_resample_slice, train_num_samples, eval_resample_slice, eval_num_samples):
        record = []

        class ImplementedPPNN(ProbabilisticProgressiveNeuralNetwork):
            def _sample_and_replace(self,
                                    resample_slice: slice):
                for network in self.networks[resample_slice]:
                    network.apply(
                        lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)
                record.append(deepcopy(self))

        ppnn = ImplementedPPNN(network, None, 'linear4', ['linear2', 'linear3'],
                               train_resample_slice, train_num_samples, eval_resample_slice, eval_num_samples)
        for _ in range(5):
            ppnn.add_new_column(True, randint(1, 10), True, True)
        return ppnn, record

    def test_ppnn(self, network):
        train_resample_slice = slice(2, 4)
        train_num_samples = randint(1, 10)
        eval_resample_slice = slice(3, 5)
        eval_num_samples = randint(1, 10)
        ppnn, record = self.ppnn_and_record(network,
                                            train_resample_slice, train_num_samples,
                                            eval_resample_slice, eval_num_samples)
        for _ in range(1):
            for train in [True, False]:
                record.clear()
                ppnn.train(train)
                x = torch.rand(5, 10)
                out = ppnn(x)
                outs_gt = [super(ProbabilisticProgressiveNeuralNetwork, pnn).forward(x) for pnn in record]
                for i, o in enumerate(out):
                    samples = torch.stack([out_gt[i] for out_gt in outs_gt], dim=1)
                    assert torch.allclose(samples, o)

                    num_samples = ppnn.train_num_samples if train else ppnn.eval_num_samples
                    assert num_samples == samples.shape[1]

                resample_slice = train_resample_slice if train else eval_resample_slice
                for i, network in enumerate(ppnn.networks):
                    if i in range(len(ppnn.networks))[resample_slice]:
                        continue
                    assert_networks_same(record[0].networks[i], network)

    def test_ppnn_full_state_dict(self, network):
        network_copy = deepcopy(network)

        train_resample_slice = slice(2, 4)
        train_num_samples = randint(1, 10)
        eval_resample_slice = slice(3, 5)
        eval_num_samples = randint(1, 10)
        ppnn, record = self.ppnn_and_record(network,
                                            train_resample_slice, train_num_samples,
                                            eval_resample_slice, eval_num_samples)

        full_state_dict = ppnn.full_state_dict()

        ppnn_copy = ProbabilisticProgressiveNeuralNetwork(network_copy)
        ppnn_copy.load_full_state_dict(full_state_dict)

        assert ppnn.train_resample_slice == ppnn_copy.train_resample_slice
        assert ppnn.train_num_samples == ppnn_copy.train_num_samples
        assert ppnn.eval_resample_slice == ppnn_copy.eval_resample_slice
        assert ppnn.eval_num_samples == ppnn_copy.eval_num_samples


class TestDropoutProgressiveNeuralNetwork:

    def test_dpnn(self, network):
        # dropout at correct positions
        # activate, deactivate dropout
        train_resample_slice = slice(2, 4)
        train_num_samples = randint(1, 10)
        eval_resample_slice = slice(3, 5)
        eval_num_samples = randint(1, 10)
        dpnn = DropoutProgressiveNeuralNetwork(network, None, 'linear4', ['linear2', 'linear3'],
                                               train_resample_slice, train_num_samples,
                                               eval_resample_slice, eval_num_samples,
                                               dropout_probability=.5, dropout_positions=['linear1', 'linear3'])
        name_to_module = dict(dpnn.base_network.named_modules(remove_duplicate=False))
        assert isinstance(name_to_module['linear1.1'], nn.Dropout)
        assert isinstance(name_to_module['linear3.1'], nn.Dropout)
        assert isinstance(name_to_module['linear3.0.0'], nn.Linear)

        for _ in range(5):
            dpnn.add_new_column(True, randint(1, 10), True, True)

        x = torch.rand(5, 10)
        dpnn_copy = deepcopy(dpnn)

        resample_slice = slice(3, 5)
        dpnn_copy._sample_and_replace(resample_slice)

        for i, (column, column_copy) in enumerate(zip(dpnn.networks, dpnn_copy.networks)):
            same_results = torch.allclose(column[-1](x), column_copy[-1](x))
            if i in range(len(dpnn.networks))[resample_slice]:
                assert not same_results
            else:
                assert same_results
