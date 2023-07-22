import torch
from torch.distributions.constraints import positive_definite

from src.curvature.utils import power_method_sum_kronecker_products_rank_1, \
    power_method_sum_kronecker_products_full_rank, make_positive_definite, \
    sum_kronecker_products
from .conftest import relative_difference

eps = torch.finfo().eps


# @pytest.mark.repeat(100)
def test_power_method_sum_kronecker_products_rank_1():
    left_tensor = torch.randn(10, 4, 5)
    right_tensor = torch.randn(10, 4, 7)
    left_opt, right_opt = power_method_sum_kronecker_products_rank_1(left_tensor, right_tensor,
                                                                     max_iter=1000, min_diff=1e-7)

    left_opt.requires_grad, right_opt.requires_grad = True, True

    ground_truth = torch.stack([torch.kron(L1.outer(L2), R1.outer(R2))
                                for left_list, right_list in zip(left_tensor, right_tensor)
                                for L1, R1 in zip(left_list, right_list)
                                for L2, R2 in zip(left_list, right_list)]).sum(dim=0)

    rdiff = relative_difference(ground_truth, torch.kron(left_opt, right_opt))

    rdiff.backward()

    assert (left_opt.grad < eps).all()
    assert (right_opt.grad < eps).all()


# @pytest.mark.repeat(100)
def test_power_method_sum_kronecker_products_full_rank():
    left_tensor = torch.randn(10, 5, 5)
    right_tensor = torch.randn(10, 7, 7)
    left_opt, right_opt = power_method_sum_kronecker_products_full_rank(left_tensor, right_tensor,
                                                                        max_iter=1000, min_diff=1e-7,
                                                                        assert_positive_definite=False)

    left_opt.requires_grad, right_opt.requires_grad = True, True

    ground_truth = torch.stack([torch.kron(L, R) for L, R in zip(left_tensor, right_tensor)]).sum(dim=0)

    rdiff = relative_difference(ground_truth, torch.kron(left_opt, right_opt))

    rdiff.backward()

    assert (left_opt.grad < eps).all()
    assert (right_opt.grad < eps).all()


# @pytest.mark.repeat(100)
def test_sum_kronecker_products():
    left_tensor = torch.randn(10, 5, 5)
    right_tensor = torch.randn(10, 7, 7)
    left_opt, right_opt = sum_kronecker_products(left_tensor, right_tensor, assert_positive_definite=False)

    left_opt.requires_grad, right_opt.requires_grad = True, True

    ground_truth = torch.stack([torch.kron(L, R) for L, R in zip(left_tensor, right_tensor)]).sum(dim=0)

    rdiff = relative_difference(ground_truth, torch.kron(left_opt, right_opt))

    rdiff.backward()

    assert (left_opt.grad < 1e-2).all()
    assert (right_opt.grad < 1e-2).all()


    left_opt2, right_opt2 = power_method_sum_kronecker_products_full_rank(left_tensor, right_tensor,
                                                                        max_iter=1000, min_diff=1e-7,
                                                                        assert_positive_definite=False)
    rdiff2 = relative_difference(ground_truth, torch.kron(left_opt2, right_opt2))

    assert rdiff < rdiff2 + eps * 2


# @pytest.mark.repeat(100)
def test_make_positive_definite():
    x = torch.randn(10, 7, 7)
    x = .5 * (x + x.transpose(-2, -1))

    x_pos = make_positive_definite(x, 1e-6)
    assert positive_definite.check(x_pos).all()
