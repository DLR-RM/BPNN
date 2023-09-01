"""Script to run and evaluate the approximate PAC-Bayes objective for Figure 6."""
import json
from os import makedirs
from os.path import join, exists

import matplotlib.pyplot as plt
import pandas as pd
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.bpnn.bpnn import *
from src.bpnn.criterions import *
from src.bpnn.curvature_scalings import *
from src.bpnn.pnn import *
from src.bpnn.utils import torch_data_path, base_path
from src.curvature.lenet5 import lenet5
from tools.mnist_dataloaders import get_train_val_test_split_dataloaders
from tools.not_mnist import NotMNIST


class PBNNTrain(BayesianProgressiveNeuralNetwork):

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
        super().__init__(
            prior=prior,
            backbone=backbone,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            train_resample_slice=train_resample_slice,
            train_num_samples=train_num_samples,
            eval_resample_slice=eval_resample_slice,
            eval_num_samples=eval_num_samples,
            weight_decay=weight_decay,
            weight_decay_layer_names=weight_decay_layer_names,
            curvature_device=curvature_device,
            curvature_scaling_device=curvature_scaling_device,
        )

    def forward(
            self,
            x: Tensor):
        return super().forward(x, num_samples=1, resample_slice=slice(-1))


def get_bounds(model, expected_empirical_risk, approx_expected_empirical_risk, len_data, confidence):
    kl_divergence = model.kl_divergence()
    approx_bound = mc_allester_bound(
        expected_empirical_risk=approx_expected_empirical_risk,
        kl_divergence=kl_divergence,
        len_data=len_data,
        confidence=confidence
    )
    gt_bound = mc_allester_bound(
        expected_empirical_risk=expected_empirical_risk,
        kl_divergence=kl_divergence,
        len_data=len_data,
        confidence=confidence
    )
    return approx_bound, gt_bound


def get_expected_empirical_risk_approx(trace, len_data, negative_data_log_likelihood, confidence, temperature_scaling):
    curvature_scaling = McAllesterScaling(
        confidence=confidence,
        fixed=(None, None, temperature_scaling),
        shared=(True, True, True))
    expected_empirical_risk = curvature_scaling._compute_expected_empirical_risk(
        trace,
        temperature_scaling=1.,
        len_data=len_data,
        negative_data_log_likelihood=negative_data_log_likelihood,
        scaled_fixed_model_size=0.
    )
    return expected_empirical_risk


def get_expected_empirical_risk_gt(model, dataloaders):
    train_metric = run_epoch(
        model, dataloaders[1][1][0], nn.CrossEntropyLoss(),
        metrics='accuracy', metrics_dim=0,
        is_classification=True, return_if_loss_nan_or_inf=True)
    if np.isnan(train_metric['loss']) or np.isinf(train_metric['loss']):
        train_metric['accuracy'] = 0.1
    expected_empirical_risk = 1 - train_metric['accuracy'] / 100
    return expected_empirical_risk


if __name__ == '__main__':
    network = lenet5(pretrained=True)

    last_layer_name = '11'
    lateral_connections = ['3', '7', '9']

    dataloaders = [(10, get_train_val_test_split_dataloaders(
        dataset_class=dataset_class,
        torch_data_path=torch_data_path,
        split=[True, False],
        transform=ToTensor(),
        batch_size=512,
        num_workers=10,
    ), CrossEntropyLoss()) for dataset_class in [MNIST, NotMNIST]]

    learning_rate = 5e-4
    confidence = 0.2
    temperature_scaling = 1e-2
    weight_decay = 1e-8

    for difference in [1., 10.]:

        pretrained_path = join(base_path, 'models', 'alpha_beta_scaling', 'state_dict.pt')

        criterion_type = MaximumAPosterioriCriterion
        curvature_scaling = McAllesterScaling(
            confidence=confidence,
            fixed=(None, None, temperature_scaling),
            shared=(True, True, True))

        base_network = deepcopy(network.to(device))

        if os.path.exists(pretrained_path):
            model = BayesianProgressiveNeuralNetwork(
                prior=KFAC(deepcopy(base_network)),
                backbone=None,
                last_layer_name=last_layer_name,
                lateral_connections=lateral_connections,
                weight_decay_layer_names=[last_layer_name],
                curvature_device=None
            )
            state_dict = torch.load(pretrained_path)
            model.load_full_state_dict(state_dict['model'])
            curvature = [KFAC(model.networks[0][0])]
            curvature[0].load_state_dict(state_dict['curvature'][0])
            negative_data_log_likelihood = state_dict['negative_data_log_likelihood']

        else:
            prior = init_prior(
                deepcopy(base_network), KFAC, dataloaders[0][1][0], weight_decay,
                StandardBayes())

            model = BayesianProgressiveNeuralNetwork(
                prior=prior,
                backbone=None,
                last_layer_name=last_layer_name,
                lateral_connections=lateral_connections,
                weight_decay_layer_names=[last_layer_name],
                curvature_device=None
            )

            criterion = criterion_type(model, confidence=confidence)
            criterion.len_data = torch.as_tensor(len(dataloaders[1][1][0].dataset))
            model.add_new_column(output_size=10)
            eval_num_samples = model.eval_num_samples
            model.eval_num_samples = 1
            train_metrics = fit(
                model, dataloaders[1][1], criterion, weight_decay=.0,
                is_classification=model.is_classification,
                learning_rate=learning_rate,
                num_epochs=100, patience=10)
            model.eval_num_samples = eval_num_samples

            curvature_scaling.reset(
                dataloader=dataloaders[1][1],
                criterion=criterion,
                is_classification=True)
            curvature, negative_data_log_likelihood = model.add_new_posterior(
                curvature_scaling, dataloaders[1][1][0],
                num_samples=1, return_curvature=True)
            state_dict = {
                'model': model.full_state_dict(),
                'curvature': [c.state_dict() for c in curvature],
                'negative_data_log_likelihood': negative_data_log_likelihood
            }
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            torch.save(state_dict, pretrained_path)
            torch.cuda.empty_cache()

        model_train = PBNNTrain(
            prior=KFAC(deepcopy(base_network)),
            backbone=None,
            last_layer_name=last_layer_name,
            lateral_connections=lateral_connections,
            weight_decay_layer_names=[last_layer_name],
            curvature_device=None
        )

        model_train.load_full_state_dict(model.full_state_dict())
        model_train.train()
        model_train.networks[0][0].load_state_dict(model_train.posterior[0][0].model_state)
        curvature_train = [KFAC(model_train.networks[0][0])]
        curvature_train[0].load_state_dict(curvature[0].state_dict())

        len_data = len(dataloaders[1][1][0].dataset)
        if negative_data_log_likelihood is None:
            negative_data_log_likelihood = run_epoch(
                model_train, dataloaders[1][1][0], nn.CrossEntropyLoss(),
                is_classification=True,
                metrics=['negative log likelihood'])[
                                               'negative log likelihood'] * len_data
        print(negative_data_log_likelihood)

        d_approx = {}
        d_gt = {}
        root_path = join(base_path, 'results', 'ablations', 'alpha_beta_scaling')
        if not exists(root_path):
            makedirs(root_path)

        a = next(iter(model.scales.values()))[0]
        b = next(iter(model.scales.values()))[1]
        print(a, b)
        assert all(
            model.scales[module][0] == a and model.scales[module][1] == b
            for module in model.networks[0][0].modules()
            if module in model.scales.keys())

        alphas = np.linspace(a - difference, a + difference, 201)
        betas = np.linspace(b - difference, b + difference, 201)

        for alpha in alphas:
            if alpha < 0:
                continue
            d_approx[alpha] = {}
            d_gt[alpha] = {}
            for beta in betas:
                if beta < 0:
                    continue
                print(alpha, beta)
                model.update_posterior(alpha, beta, curvature)
                model.update_scaling(alpha=alpha, beta=beta, temperature_scaling=temperature_scaling)
                model_train.update_posterior(alpha, beta, curvature_train)
                model_train.update_scaling(alpha=alpha, beta=beta, temperature_scaling=temperature_scaling)

                expected_empirical_risk = get_expected_empirical_risk_gt(model, dataloaders)
                approx_expected_empirical_risk = get_expected_empirical_risk_approx(
                    lambda _: model_train.trace(curvature_train),
                    len_data, negative_data_log_likelihood, confidence,
                    temperature_scaling)
                approx_bound, gt_bound = get_bounds(
                    model_train, expected_empirical_risk, approx_expected_empirical_risk,
                    len_data, confidence)
                d_approx[alpha][beta] = approx_bound.item()
                d_gt[alpha][beta] = gt_bound.item()

        with open(os.path.join(root_path, f'alpha_beta_scaling_approx_{difference}.json'), 'w') as f:
            json.dump(d_approx, f, indent=4, default=str)

        with open(os.path.join(root_path, f'alpha_beta_scaling_gt_{difference}.json'), 'w') as f:
            json.dump(d_gt, f, indent=4, default=str)


    def get_alpha_beta_values(name):
        approx_json = json.load(open(os.path.join(root_path, name)))
        approx = pd.DataFrame(approx_json)
        approx.columns.name = 'alpha'
        approx.index.name = 'beta'
        approx.columns = approx.columns.astype(float)
        approx.index = approx.index.astype(float)
        abv = [(alpha, beta, value) for alpha, d in approx_json.items() for beta, value in d.items()]
        return np.array(list(zip(*abv)), dtype=float)


    def plot(alpha, beta, values):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(alpha, beta, values, cmap=plt.cm.viridis)
        ax.set_zlim(0, 2)


    def plot_distances(distances):
        alpha_approx, beta_approx, values_approx = np.concatenate(
            [get_alpha_beta_values(f'alpha_beta_scaling_approx_{distance}.json')
             for distance in distances], axis=1)
        alpha_gt, beta_gt, values_gt = np.concatenate(
            [get_alpha_beta_values(f'alpha_beta_scaling_gt_{distance}.json')
             for distance in distances], axis=1)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(alpha_approx, beta_approx, values_approx)
        ax.plot_trisurf(alpha_gt, beta_gt, values_gt)
        ax.set_zlim(0, 2)
        ax.view_init(elev=20., azim=-69)
        target_path = join(base_path, 'reports', 'ablations', 'alpha_beta.png')
        if not exists(os.path.dirname(target_path)):
            makedirs(os.path.dirname(target_path))
        fig.savefig(target_path)
        plt.show()


    plot_distances([1.0, 10.])
