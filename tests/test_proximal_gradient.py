import unittest
from typing import TypeVar, cast

import jax
import numpy as np
from jaxopt.prox import prox_lasso
from numpy.testing import assert_array_almost_equal
from numpy.typing import NBitBase, NDArray

from zfista import minimize_proximal_gradient
from zfista._typing import FloatArray

T = TypeVar("T", bound=NBitBase)

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


def build_dataset(
    n_samples: int = 50,
    n_features: int = 200,
    n_informative_features: int = 10,
    n_targets: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    build an ill-posed linear regression problem with many noisy features and
    comparatively few samples
    See https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py
    """
    random_state = np.random.RandomState(0)
    if n_targets > 1:
        w = random_state.randn(n_features, n_targets)
    else:
        w = random_state.randn(n_features)
    w[n_informative_features:] = 0.0
    X = random_state.randn(n_samples, n_features)
    y = np.dot(X, w)
    X_test = random_state.randn(n_samples, n_features)
    y_test = np.dot(X_test, w)
    return X, y, X_test, y_test


class TestProximalGradient(unittest.TestCase):
    def test_minimize_proximal_gradient_lasso_zero(self) -> None:
        A = np.array([[0], [0], [0]])
        b = np.array([0, 0, 0])
        x0 = np.random.random(1)
        l1_ratio = 0.1

        def f(x: NDArray[np.floating[T]]) -> np.floating[T]:
            return np.linalg.norm(A @ x - b) ** 2 / 6

        def g(x: NDArray[np.floating[T]]) -> np.floating[T]:
            return l1_ratio * np.linalg.norm(x, ord=1)

        def jac_f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
            return cast(NDArray[np.floating[T]], A.T @ (A @ x - b) / 3)

        def prox_wsum_g(
            weight: NDArray[np.floating[T]], x: NDArray[np.floating[T]]
        ) -> NDArray[np.floating[T]]:
            return cast(NDArray[np.floating[T]], prox_lasso(x, l1_ratio * weight))

        res = minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0)
        res_nesterov = minimize_proximal_gradient(
            f, g, jac_f, prox_wsum_g, x0, nesterov=True
        )
        assert_array_almost_equal(res.x, [0], decimal=3)
        assert_array_almost_equal(res_nesterov.x, [0], decimal=3)

    def test_minimize_proximal_gradient_lasso_toy(self) -> None:
        """
        min (1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1
        See https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py
        """
        A = np.array([[-1], [0], [1]])
        b = np.array([-1, 0, 1])
        x0 = np.random.random(1)
        l1_ratios = [1e-8, 0.1, 0.5, 1]
        for l1_ratio in l1_ratios:

            def f(x: NDArray[np.floating[T]]) -> np.floating[T]:
                return np.linalg.norm(A @ x - b) ** 2 / 6

            def g(
                x: NDArray[np.floating[T]], l1_ratio: float = l1_ratio
            ) -> np.floating[T]:
                return l1_ratio * np.linalg.norm(x, ord=1)

            def jac_f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
                return cast(NDArray[np.floating[T]], A.T @ (A @ x - b) / 3)

            def prox_wsum_g(
                weight: NDArray[np.floating[T]],
                x: NDArray[np.floating[T]],
                l1_ratio: float = l1_ratio,
            ) -> NDArray[np.floating[T]]:
                return cast(NDArray[np.floating[T]], prox_lasso(x, l1_ratio * weight))

            res = minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0)
            res_nesterov = minimize_proximal_gradient(
                f, g, jac_f, prox_wsum_g, x0, nesterov=True
            )
            if l1_ratio == 1e-8:
                assert_array_almost_equal(res.x, [1], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [1], decimal=3)
            if l1_ratio == 0.1:
                assert_array_almost_equal(res.x, [0.85], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.85], decimal=3)
            if l1_ratio == 0.5:
                assert_array_almost_equal(res.x, [0.25], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.25], decimal=3)
            if l1_ratio == 1:
                assert_array_almost_equal(res.x, [0], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0], decimal=3)

    def test_minimize_proximal_gradient_biobjective_lasso_toy(self) -> None:
        """
        min ((1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1,
            (1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1)
        See https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py
        """
        A = np.array([[-1], [0], [1]])
        b = np.array([-1, 0, 1])
        x0 = np.random.random(1)
        l1_ratios = [1e-8, 0.1, 0.5, 1]
        for l1_ratio in l1_ratios:

            def f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
                val = np.linalg.norm(A @ x - b) ** 2 / 6
                return np.array([val, val])

            def g(
                x: NDArray[np.floating[T]], l1_ratio: float = l1_ratio
            ) -> NDArray[np.floating[T]]:
                val = l1_ratio * np.linalg.norm(x, ord=1)
                return np.array([val, val])

            def jac_f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
                grad_fi = A.T @ (A @ x - b) / 3
                return np.vstack([grad_fi, grad_fi])

            def prox_wsum_g(
                weight: NDArray[np.floating[T]],
                x: NDArray[np.floating[T]],
                l1_ratio: float = l1_ratio,
            ) -> NDArray[np.floating[T]]:
                return cast(
                    NDArray[np.floating[T]], prox_lasso(x, l1_ratio * weight.sum())
                )

            res = minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0)
            res_nesterov = minimize_proximal_gradient(
                f, g, jac_f, prox_wsum_g, x0, nesterov=True
            )
            if l1_ratio == 1e-8:
                assert_array_almost_equal(res.x, [1], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [1], decimal=3)
            if l1_ratio == 0.1:
                assert_array_almost_equal(res.x, [0.85], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.85], decimal=3)
            if l1_ratio == 0.5:
                assert_array_almost_equal(res.x, [0.25], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.25], decimal=3)
            if l1_ratio == 1:
                assert_array_almost_equal(res.x, [0], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0], decimal=3)

    def test_minimize_proximal_gradient_triobjective_lasso_toy(self) -> None:
        """
        min ((1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1,
            (1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1,
            (1 / 2) ||Ax - b||^2 + l1_ratio * ||x||_1)
        See https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py
        """
        A = np.array([[-1], [0], [1]])
        b = np.array([-1, 0, 1])
        x0 = np.random.random(1)
        l1_ratios = [1e-8, 0.1, 0.5, 1]
        for l1_ratio in l1_ratios:

            def f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
                val = np.linalg.norm(A @ x - b) ** 2 / 6
                return np.array([val, val, val])

            def g(
                x: NDArray[np.floating[T]], l1_ratio: float = l1_ratio
            ) -> NDArray[np.floating[T]]:
                val = l1_ratio * np.linalg.norm(x, ord=1)
                return np.array([val, val, val])

            def jac_f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
                grad_fi = A.T @ (A @ x - b) / 3
                return np.vstack([grad_fi, grad_fi, grad_fi])

            def prox_wsum_g(
                weight: NDArray[np.floating[T]],
                x: NDArray[np.floating[T]],
                l1_ratio: float = l1_ratio,
            ) -> NDArray[np.floating[T]]:
                return cast(
                    NDArray[np.floating[T]], prox_lasso(x, l1_ratio * weight.sum())
                )

            res = minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0)
            res_nesterov = minimize_proximal_gradient(
                f, g, jac_f, prox_wsum_g, x0, nesterov=True
            )
            if l1_ratio == 1e-8:
                assert_array_almost_equal(res.x, [1], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [1], decimal=3)
            if l1_ratio == 0.1:
                assert_array_almost_equal(res.x, [0.85], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.85], decimal=3)
            if l1_ratio == 0.5:
                assert_array_almost_equal(res.x, [0.25], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0.25], decimal=3)
            if l1_ratio == 1:
                assert_array_almost_equal(res.x, [0], decimal=3)
                assert_array_almost_equal(res_nesterov.x, [0], decimal=3)

    def test_minimize_proximal_gradient_return_all(self) -> None:
        A = np.array([[0], [0], [0]])
        b = np.array([0, 0, 0])
        x0 = np.random.random(1)
        l1_ratio = 0.1

        def f(x: NDArray[np.floating[T]]) -> np.floating[T]:
            return np.linalg.norm(A @ x - b) ** 2 / 6

        def g(x: NDArray[np.floating[T]]) -> np.floating[T]:
            return l1_ratio * np.linalg.norm(x, ord=1)

        def jac_f(x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
            return cast(NDArray[np.floating[T]], A.T @ (A @ x - b) / 3)

        def prox_wsum_g(
            weight: NDArray[np.floating[T]], x: NDArray[np.floating[T]]
        ) -> NDArray[np.floating[T]]:
            return cast(NDArray[np.floating[T]], prox_lasso(x, l1_ratio * weight))

        res = minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0, return_all=True)
        assert "allvecs" in res
        assert "allerrs" in res
