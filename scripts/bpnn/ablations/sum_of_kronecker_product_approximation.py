"""Script to run and evaluate the quality of the power method for approximating
sums of Kronecker products for Figure 3."""
from math import sqrt

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt


def relative_difference(
        ground_truth: torch.Tensor,
        other: torch.tensor,
        ord=None,
        dim=None) -> torch.Tensor:
    """Computes the relative difference of the two tensors."""
    return torch.linalg.norm(ground_truth - other, ord=ord, dim=dim) \
        / (torch.linalg.norm(ground_truth, ord=ord, dim=dim) + torch.finfo(ground_truth.dtype).eps)


def power_method_sum_kronecker_products_full_rank(
        left_tensor: torch.Tensor,
        right_tensor: torch.Tensor,
        max_iter: int = 100):
    r"""Approximation of sums of general Kronecker products with the power method.

    It approximates
        .. math:
            L, R = \text{arg min}_{L, R} \| \frac{1}{B} \sum_{i = 1}^B
                left_tensor[i] \otimes right_tensor[i] - L \otimes R\|

    It corresponds to Algorithm 1 of
    `Kronecker-Factored Optimal Curvature <http://bayesiandeeplearning.org/2021/papers/33.pdf>_`

    Args:
        left_tensor: (shape [B, M, M]) The first tensor
        right_tensor: (shape [B, N, N]) The second tensor
        max_iter: Maximal number of iterations

    Returns:
        The two tensors L (shape [M, M]) and R (shape [N, N])
    """
    L_opts = []
    R_opts = []

    M = left_tensor.shape[2]
    L_opt = torch.randn(M, M, dtype=left_tensor.dtype, device=left_tensor.device)
    tiny = torch.finfo(L_opt.dtype).tiny

    # make L_opt positive definite such that L_opt, R_opt are positive semidefinite
    L_opt = L_opt @ L_opt.T + torch.eye(M, dtype=left_tensor.dtype, device=left_tensor.device)
    L_opt /= L_opt.norm() + tiny
    L_opts.append(L_opt)
    for i in range(max_iter):
        R_opt = torch.tensordot(torch.tensordot(L_opt, left_tensor, dims=[[0, 1], [1, 2]]), right_tensor, dims=1)
        R_opts.append(R_opt)
        R_opt = R_opt / (R_opt.norm() + tiny)
        L_opt_n = torch.tensordot(torch.tensordot(R_opt, right_tensor, dims=[[0, 1], [1, 2]]), left_tensor, dims=1)
        L_opt_n = L_opt_n / (L_opt_n.norm() + tiny)
        L_opts.append(L_opt_n)
        L_opt = L_opt_n
    R_opt = torch.tensordot(torch.tensordot(L_opt, left_tensor, dims=[[0, 1], [1, 2]]), right_tensor, dims=1)
    R_opts.append(R_opt)
    return L_opts, R_opts


def generate_matrices(size=10, scale=1., weight_decay=1e-4, num_matrices=2, eps=1e-8):
    matrices = sqrt(scale) * torch.randn(num_matrices - 1, size, size)
    matrices = matrices @ matrices.transpose(-1, -2) + eps * torch.eye(size)[None, :, :]
    matrices = torch.cumsum(matrices * torch.arange(1, num_matrices)[:, None, None], dim=0)
    weight_decay = sqrt(weight_decay) * torch.eye(size)[None, :, :]
    matrices = torch.cat([weight_decay, matrices], dim=0)
    return matrices


def test_approximation_quality(size=10, scale=1., weight_decay=1e-4, num_matrices=2, eps=1e-8):
    left_matrices = generate_matrices(
        size=size, scale=scale, weight_decay=weight_decay, num_matrices=num_matrices, eps=eps)

    right_matrices = generate_matrices(
        size=size, scale=scale, weight_decay=weight_decay, num_matrices=num_matrices, eps=eps)

    L_opts, R_opts = power_method_sum_kronecker_products_full_rank(left_matrices, right_matrices)
    L_opt, R_opt = L_opts[-1], R_opts[-1]

    power_method_solution = torch.kron(L_opt, R_opt)
    sum_solution = torch.kron(left_matrices.sum(dim=0), right_matrices.sum(dim=0))
    ground_truth = sum(torch.kron(L, R) for L, R in zip(left_matrices, right_matrices))

    power_method_error = relative_difference(ground_truth, power_method_solution)
    sum_error = relative_difference(ground_truth, sum_solution)
    return power_method_error.item(), sum_error.item()


if __name__ == '__main__':
    # Figure 3 (a)
    B, M, N = 3, 50, 70

    num_experiments = 100
    max_iter = 10

    results = torch.zeros(num_experiments, max_iter)
    results_sum = torch.zeros(num_experiments)
    results_mean = torch.zeros(num_experiments)
    for i in range(num_experiments):
        left_tensor = generate_matrices(size=M, num_matrices=B + 1, weight_decay=0.)
        right_tensor = generate_matrices(size=N, num_matrices=B + 1, weight_decay=0.)

        ground_truth = sum(torch.kron(L, R) for L, R in zip(left_tensor, right_tensor))
        results_sum[i] = relative_difference(ground_truth, torch.kron(left_tensor.sum(dim=0), right_tensor.sum(dim=0)))
        results_mean[i] = relative_difference(
            ground_truth,
            torch.kron(left_tensor.mean(dim=0), right_tensor.sum(dim=0)))

        left_approxs, right_approxs = power_method_sum_kronecker_products_full_rank(
            left_tensor,
            right_tensor,
            max_iter=max_iter - 1)
        others = [torch.kron(L, R) for L, R in zip(left_approxs, right_approxs)]
        for j, other in enumerate(others):
            results[i, j] = relative_difference(ground_truth, other)

    print(results.mean(dim=0))
    print(results.std(dim=0))

    # Figure 3 (b) (i)
    sizes = range(2, 20)
    results = [
        list(test_approximation_quality(size=size, num_matrices=2)) + [size, run]
        for size in sizes for run in range(num_experiments)]
    df = pd.DataFrame(results, columns=['Power method', 'Sum', 'size', 'run'])
    sns.lineplot(data=df, x='size', y='Power method')
    sns.lineplot(data=df, x='size', y='Sum')
    plt.ylabel('Relative error')
    plt.loglog()
    plt.show()

    # Figure 3 (b) (ii)
    scales = [10 ** i for i in range(-5, 5)]
    results = [
        list(test_approximation_quality(scale=scale, num_matrices=2)) + [scale, run]
        for scale in scales for run in range(num_experiments)]
    df = pd.DataFrame(results, columns=['Power method', 'Sum', 'scale', 'run'])
    sns.lineplot(data=df, x='scale', y='Power method')
    sns.lineplot(data=df, x='scale', y='Sum')
    plt.ylabel('Relative error')
    plt.loglog()
    plt.show()

    # Figure 3 (b) (iii)
    weight_decays = [10 ** i for i in range(-10, 1)]
    results = [
        list(test_approximation_quality(weight_decay=weight_decay, num_matrices=2)) + [weight_decay, run]
        for weight_decay in weight_decays for run in range(num_experiments)]
    df = pd.DataFrame(results, columns=['Power method', 'Sum', 'weight decay', 'run'])
    sns.lineplot(data=df, x='weight decay', y='Power method')
    sns.lineplot(data=df, x='weight decay', y='Sum')
    plt.ylabel('Relative error')
    plt.loglog()
    plt.show()

    # Figure 3 (b) (iv)
    num_matrices = range(2, 10)
    results = [
        list(test_approximation_quality(num_matrices=n + 1, weight_decay=0.)) + [n, run]
        for n in num_matrices for run in range(num_experiments)]
    df = pd.DataFrame(results, columns=['Power method', 'Sum', 'num matrices', 'run'])
    sns.lineplot(data=df, x='num matrices', y='Power method')
    sns.lineplot(data=df, x='num matrices', y='Sum')
    plt.ylabel('Relative error')
    plt.loglog()
    plt.show()
