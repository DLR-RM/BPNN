"""Provides utility functions except for plotting which are in `plot.py`."""

import logging
import os
import random
from datetime import datetime
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm


def get_eigenvalues(
        factors: List[Tensor],
        verbose: bool = False) -> Tensor:
    """Computes the eigenvalues of KFAC, EFB or diagonal factors.

    Args:
        factors: A list of KFAC, EFB or diagonal factors.
        verbose: Prints out progress if True.

    Returns:
        The eigenvalues of all KFAC, EFB or diagonal factors.
    """
    eigenvalues = Tensor()
    factors = tqdm(factors, disable=not verbose)
    for layer, factor in enumerate(factors):
        factors.set_description(desc=f"Layer [{layer + 1}/{len(factors)}]")
        if len(factor) == 2:
            xxt_eigvals = torch.symeig(factor[0])[0]
            ggt_eigvals = torch.symeig(factor[1])[0]
            eigenvalues = torch.cat([eigenvalues, torch.ger(xxt_eigvals, ggt_eigvals).contiguous().view(-1)])
        else:
            eigenvalues = torch.cat([eigenvalues, factor.contiguous().view(-1)])
    return eigenvalues


def get_eigenvectors(factors: Dict[Module, Tensor]) -> Dict[Module, Tensor]:
    """Computes the eigenvectors of KFAC factors.

    Args:
        factors: A dict mapping layers to lists of first and second KFAC factors.

    Returns:
        A dict mapping layers to lists containing the first and second KFAC factors eigenvectors.
    """
    eigenvectors = dict()
    for layer, (xxt, ggt) in factors.items():
        sym_xxt, sym_ggt = xxt + xxt.t(), ggt + ggt.t()
        _, xxt_eigvecs = torch.symeig(sym_xxt, eigenvectors=True)
        _, ggt_eigvecs = torch.symeig(sym_ggt, eigenvectors=True)
        eigenvectors[layer] = (xxt_eigvecs, ggt_eigvecs)
    return eigenvectors


def power_method_sum_kronecker_products_rank_1(
        left_tensor: Tensor,
        right_tensor: Tensor,
        max_iter: int = 100,
        min_diff: float = 1e-5,
        dtype: torch.dtype = None) -> Tuple[Tensor, Tensor]:
    r"""Approximation of sums of outer-product Kronecker products with the power method.

    It approximates
        .. math:
            L, R = \text{arg min}_{L, R} \| \frac{1}{B} \sum_{i = 1}^B \sum_{t = 1}^T \sum_{t' = 1}^T
                left_tensor[i, t].outer(left_tensor[i, t']) \otimes right_tensor[i, t].outer(right_tensor[i, t'])
                - L \otimes R\|

    It corresponds to Algorithm 2 of
    `Kronecker-Factored Optimal Curvature <http://bayesiandeeplearning.org/2021/papers/33.pdf>_`

    Args:
        left_tensor: (shape [B, T, M]) The first tensor
        right_tensor: (shape [B, T, N]) The second tensor
        max_iter: Maximal number of iterations
        min_diff: If the difference between two updates of the left factor is
            smaller than this, the algorithm stops
        dtype: The data type

    Returns:
        The two tensors L (shape [M, M]) and R (shape [N, N])
    """
    if dtype is None:
        dtype = left_tensor.dtype
    tiny = torch.finfo(dtype).tiny
    M = left_tensor.shape[2]
    L_opt = torch.randn(M, M, dtype=dtype, device=left_tensor.device)

    # make L_opt positive definite such that L_opt, R_opt are positive semidefinite
    L_opt = L_opt @ L_opt.T + torch.eye(M, dtype=dtype, device=left_tensor.device)
    L_opt /= L_opt.norm() + tiny
    RL = (right_tensor.transpose(1, 2) @ left_tensor).to(dtype)
    for i in range(max_iter):
        R_opt = torch.tensordot((RL @ L_opt), RL, dims=[[0, 2], [0, 2]])
        R_opt /= R_opt.norm() + tiny
        L_opt_n = torch.tensordot(torch.tensordot(RL, R_opt, dims=[[1, ], [0, ]]), RL, dims=[[0, 2], [0, 1]])
        L_opt_n /= L_opt_n.norm() + tiny
        diff = (L_opt - L_opt_n).norm()
        if diff < min_diff:
            break
        L_opt = L_opt_n
    R_opt = torch.tensordot((RL @ L_opt), RL, dims=[[0, 2], [0, 2]])
    return L_opt.to(left_tensor.dtype), R_opt.to(right_tensor.dtype)


def power_method_sum_kronecker_products_full_rank(
        left_tensor: Tensor,
        right_tensor: Tensor,
        assert_positive_definite: bool = True,
        max_iter: int = 100,
        min_diff: float = 1e-6) -> Tuple[Tensor, Tensor]:
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
        assert_positive_definite: If the result should be made positive definite
            to counteract numeric instabilities
        max_iter: Maximal number of iterations
        min_diff: If the difference between two updates of the left factor is
            smaller than this, the algorithm stops

    Returns:
        The two tensors L (shape [M, M]) and R (shape [N, N])
    """
    M = left_tensor.shape[2]
    L_opt = torch.randn(M, M, dtype=left_tensor.dtype, device=left_tensor.device)
    tiny = torch.finfo(L_opt.dtype).tiny

    # make L_opt positive definite such that L_opt, R_opt are positive semidefinite
    L_opt = L_opt @ L_opt.T + torch.eye(M, dtype=left_tensor.dtype, device=left_tensor.device)
    L_opt /= L_opt.norm() + tiny
    for i in range(max_iter):
        R_opt = torch.tensordot(torch.tensordot(L_opt, left_tensor, dims=[[0, 1], [1, 2]]), right_tensor, dims=1)
        R_opt = R_opt / (R_opt.norm() + tiny)
        L_opt_n = torch.tensordot(torch.tensordot(R_opt, right_tensor, dims=[[0, 1], [1, 2]]), left_tensor, dims=1)
        L_opt_n = L_opt_n / (L_opt_n.norm() + tiny)
        diff = (L_opt - L_opt_n).norm()
        if diff < min_diff:
            break
        L_opt = L_opt_n
    R_opt = torch.tensordot(torch.tensordot(L_opt, left_tensor, dims=[[0, 1], [1, 2]]), right_tensor, dims=1)
    if assert_positive_definite:
        L_opt, R_opt = check_and_make_pd(L_opt), check_and_make_pd(R_opt)
    return L_opt, R_opt


def sum_kronecker_products(
        left_tensor: Tensor,
        right_tensor: Tensor,
        assert_positive_definite: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Approximation of sums of general Kronecker products with the power method.

    It approximates
        .. math:
            L, R = \text{arg min}_{L, R} \| \frac{1}{B} \sum_{i = 1}^B
                left_tensor[i] \otimes right_tensor[i] - L \otimes R\|

    It corresponds to a generalized version of Algorithm 2 of the appendix of
    `Natural continual learning: success is a journey, not (just) a destination
    <https://proceedings.neurips.cc/paper/2021/file/ec5aa0b7846082a2415f0902f0da88f2-Paper.pdf>_`

    Args:
        left_tensor: (shape [B, M, M]) The first tensor
        right_tensor: (shape [B, N, N]) The second tensor
        assert_positive_definite: If the result should be made positive definite
            to counteract numeric instabilities

    Returns:
        The two tensors L (shape [M, M]) and R (shape [N, N])
    """
    l = left_tensor.view(left_tensor.shape[0], -1)
    r = right_tensor.view(right_tensor.shape[0], -1)
    Q, _ = torch.linalg.qr(l.T)
    H = (l @ Q).T @ r
    U, s, VT = torch.linalg.svd(H, full_matrices=False)
    y = s[0].sqrt() * VT[0]
    x = s[0].sqrt() * Q @ U[:, 0]
    L, R = x.view(*left_tensor.shape[1:]), y.view(*right_tensor.shape[1:])
    if torch.trace(L) < 0 and torch.trace(R) < 0:
        L, R = -L, -R
    if assert_positive_definite:
        L, R = check_and_make_pd(L), check_and_make_pd(R)
    return L, R


def is_positive_definite(x: Tensor) -> bool:
    """Checks if the tensor is positive definite."""
    L, info = torch.linalg.cholesky_ex(x)
    return info == 0 and L.isfinite().all()


def make_positive_definite2(
        x: Tensor,
        eps: Optional[float] = None) -> Tensor:
    """Makes the input positive definite."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    x_new = x + eps * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return x_new


def make_positive_definite(
        x: Tensor,
        eps: Optional[float] = None) -> Tensor:
    """Makes the input positive definite."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    try:
        eigval, eigvec = torch.linalg.eigh(x)
        eigval = torch.maximum(eigval, torch.as_tensor(eps).to(eigval))
        return eigvec @ eigval.diag_embed() @ eigvec.transpose(-2, -1)
    except RuntimeError:
        print("PyTorch eigh does not converge. Adding eps to diagonal.")
        return make_positive_definite2(x, eps)


def check_and_make_pd(
        x: Tensor,
        eps: Optional[float] = None) -> Tensor:
    """Increases eps until the tensor is positive definite."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    x_sym = .5 * (x + x.transpose(-2, -1))
    x_sym = make_positive_definite2(x_sym, eps)
    out = x_sym.clone()
    while not is_positive_definite(out):
        out = make_positive_definite(x_sym, eps)
        eps *= 10
    return out


def make_invertible_and_invert(
        x: Tensor,
        eps: Optional[float] = None) -> Tensor:
    """Increases eps until the tensor is invertible."""
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    if not x.isfinite().all():
        x[x.isnan()] = 0
        x[x.isinf()] = torch.finfo(torch.float).max
    while True:
        inv, info = torch.linalg.inv_ex(x + eps * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device))
        if info == 0 and inv.isfinite().all():
            return inv
        eps *= 10
        if eps > 1e-5:
            print("inv eps = {}".format(eps))


def invert_and_cholesky(x: Tensor) -> Tensor:
    """Inverts and Cholesky decomposes the input."""
    inv, info = torch.linalg.inv_ex(x)
    if info != 0 or not inv.isfinite().all():
        inv = make_invertible_and_invert(x)

    inv_chol, info = torch.linalg.cholesky_ex(inv)
    if info != 0 or not inv_chol.isfinite().all():
        inv = check_and_make_pd(inv)
        inv_chol = torch.linalg.cholesky(inv)
    return inv_chol


def seed_all_rng(seed: Union[int, None] = None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed: The seed value to use. If None, will use a strong random seed.
    """
    if seed is None:
        seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    return seed
