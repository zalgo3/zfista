from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, cast

import jax
import numpy as np
from jaxopt.projection import projection_box
from jaxopt.prox import prox_lasso
from numpy.typing import NBitBase, NDArray

from zfista import minimize_proximal_gradient
from zfista._typing import FloatArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.optimize import OptimizeResult

T = TypeVar("T", bound=NBitBase)

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]


class Problem:
    """Superclass of test problems to be solved by the proximal gradient methods for
    multiobjective optimization.

    In all test problems, each objective function can be written as

    .. math::

        F_i(x) = f_i(x) + g_i(x),

    where :math:`f_i` is convex and differentiable
    and :math:`g_i` is closed, proper and convex.

    Parameters
    ----------
    n_features
        The dimension of the decision variable.

    n_objectives
        The number of objective functions.

    l1_ratios
        An array of shape (n_objectives,) containing the coefficients
        for the L1 regularization term for each objective function.
        If not provided, no L1 regularization is applied.

    l1_shifts
        An array of shape (n_objectives,) containing the shifts
        for the L1 regularization term for each objective function.
        If not provided, no shifts are applied.

    bounds
        A tuple with two elements representing the lower and upper bounds
        of the decision variable.
        Each element can be a scalar or an array of shape (n_features,).
        If not provided, no bounds are applied.
    """

    def __init__(
        self,
        n_features: int,
        n_objectives: int,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[NDArray[np.floating[T]] | float, NDArray[np.floating[T]] | float]
        | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_objectives = n_objectives
        self.l1_ratios = None if l1_ratios is None else np.array(l1_ratios)
        self.l1_shifts = (
            np.zeros(n_objectives) if l1_shifts is None else np.array(l1_shifts)
        )
        self.bounds = bounds
        self.name = self._generate_name()

    def _generate_name(self) -> str:
        name_parts = [type(self).__name__, f"n_{self.n_features}"]
        if self.l1_ratios is not None:
            l1_ratios_str = "_".join(map(str, self.l1_ratios))
            name_parts.append(f"l1_ratios_{l1_ratios_str}")
            l1_shifts_str = "_".join(map(str, self.l1_shifts))
            name_parts.append(f"l1_shifts_{l1_shifts_str}")
        if self.bounds is not None:
            bounds_str = "_".join(map(str, [self.bounds[0], self.bounds[1]]))
            name_parts.append(f"bounds_{bounds_str}")
        return "_".join(name_parts)

    @abstractmethod
    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        pass

    @abstractmethod
    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        pass

    def g(self, x: FloatArray) -> FloatArray:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        if self.bounds is not None:
            if (x < self.bounds[0]).any() or (x > self.bounds[1]).any():
                return np.full(self.n_objectives, np.inf)
        if self.l1_ratios is not None:
            if self.n_objectives != len(self.l1_ratios):
                raise ValueError("len(l1_ratios) should be equal to n_objectives.")
            if self.n_objectives != len(self.l1_shifts):
                raise ValueError("len(l1_shifts) should be equal to n_objectives.")
            return cast(
                FloatArray,
                self.l1_ratios
                * np.linalg.norm(x - self.l1_shifts.reshape(-1, 1), ord=1, axis=1),
            )
        return np.zeros(self.n_objectives)

    def prox_wsum_g(
        self, weight: NDArray[np.floating[T]], x: NDArray[np.floating[T]]
    ) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        if self.n_objectives != len(weight):
            raise ValueError("len(weight) should be equal to n_objectives.")
        if self.l1_ratios is not None:
            coef = weight * self.l1_ratios
            x = prox_lasso(
                x + np.sum(coef[1:]) - self.l1_shifts[0] + self.l1_shifts[0], coef[0]
            )
            for i in range(1, self.n_objectives):
                x = (
                    prox_lasso(x - coef[i] - self.l1_shifts[i], coef[i])
                    + self.l1_shifts[i]
                )
        if self.bounds is not None:
            x = projection_box(x, (self.bounds[0], self.bounds[1]))
        return x

    def minimize_proximal_gradient(
        self, x0: NDArray[np.floating[T]], **kwargs: Any
    ) -> OptimizeResult:
        return minimize_proximal_gradient(
            self.f,
            self.g,
            self.jac_f,
            self.prox_wsum_g,
            x0,
            **kwargs,
        )


class JOS1(Problem):
    r"""n_features = 5 (default), n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = (1 / n) \sum_i x_i^2, \\
        f_2(x) = (1 / n) \sum_i (x_i - 2)^2.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \nabla f_1(x) = (2 / n) x, \nabla f_2(x) = (2 / n) (x - 2).

    Reference: Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for
    evolutionary multi-objective optimization: Why does it work and how?
    In: GECCO’01 Proceedings of the 3rd Annual Conference on Genetic and Evolutionary
    Computation, pp. 1042–1049 (2001)
    """

    def __init__(
        self,
        n_features: int = 5,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[NDArray[np.floating[T]] | float, NDArray[np.floating[T]] | float]
        | None = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = np.linalg.norm(x) ** 2 / self.n_features
        f2 = np.linalg.norm(x - 2) ** 2 / self.n_features
        return np.array([f1, f2])

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = 2 * x / self.n_features
        jac_f2 = 2 * (x - 2) / self.n_features
        return np.vstack((jac_f1, jac_f2))


class SD(Problem):
    r"""n_features = 4, n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = 2 x_1 + \sqrt{2} x_2 + \sqrt{2} x_3 + x_4, \\
        f_2(x) = 2 / x_1 + 2 \sqrt{2} / x_2 + 2 \sqrt{2} / x_3 + x_4,
        \end{gathered}

    subject to

    .. math::

        [1, \sqrt{2}, \sqrt{2}, 1] \le x \le [3, 3, 3, 3].

    Each gradient of f_i can be written as

    .. math::

        \nabla f_1(x) = [1, \sqrt{2}, \sqrt{2}, 1], \nabla f_2(x) = 0.

    Reference: Stadler, W., Dauer, J.: Multicriteria optimization in engineering:
    a tutorial and survey. In: Kamat, M.P. (ed.) Progress in Aeronautics and
    Astronautics: Structural Optimization: Status and Promise, vol. 150, pp. 209–249.
    American Institute of Aeronautics and Astronautics, Reston (1992)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(
            n_features=4,
            n_objectives=2,
            bounds=(1e-6, np.inf),
        )

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = 2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(2) * x[2] + x[3]
        f2 = 2 / x[0] + 2 * np.sqrt(2) / x[1] + 2 * np.sqrt(2) / x[2] + 2 / x[3]
        return np.array([f1, f2])

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        jac_f1 = np.array([2, np.sqrt(2), np.sqrt(2), 1])
        jac_f2 = np.array(
            [
                -2 / x[0] ** 2,
                -2 * np.sqrt(2) / x[1] ** 2,
                -2 * np.sqrt(2) / x[2] ** 2,
                -2 / x[3] ** 2,
            ]
        )
        return np.vstack((jac_f1, jac_f2))


class FDS(Problem):
    r"""n_features = 10 (default), n_objectives = 3

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = \sum_i i (x_i - i)^4 / n^2, \\
        f_2(x) = \exp(\sum_i x_i / n) + \|x\|^2, \\
        f_3(x) = \sum_i i (n - i + 1) \exp(-x_i) / (n (n + 1)).
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = 4 / n^2 \sum_i i (x_i - i)^3, \\
        \nabla f_2(x) = \exp(\sum_i x_i / n) / n + 2 x, \\
        \nabla f_3(x) = - [i (n - i + 1) \exp(-x_i) / (n (n + 1))]_i
        \end{gathered}

    Reference: Fliege, J., Graña Drummond, L.M., Svaiter, B.F.: Newton’s method for
    multiobjective optimization. SIAM J. Optim. 20(2), 602–626 (2009)
    """

    def __init__(
        self,
        n_features: int = 10,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[NDArray[np.floating[T]] | float, NDArray[np.floating[T]] | float]
        | None = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )
        self.one_to_n = np.arange(self.n_features) + 1
        self.conv_n = self.one_to_n * self.one_to_n[::-1]

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = np.inner(self.one_to_n, (x - self.one_to_n) ** 4) / self.n_features**2
        f2 = np.exp(x.sum() / self.n_features) + np.linalg.norm(x) ** 2
        f3 = np.inner(self.conv_n, np.exp(-x)) / (
            self.n_features * (self.n_features + 1)
        )
        return np.array([f1, f2, f3])

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = 4 / self.n_features**2 * self.one_to_n * (x - self.one_to_n) ** 3
        jac_f2 = np.exp(x.sum() / self.n_features) / self.n_features + 2 * x
        jac_f3 = -self.conv_n * np.exp(-x) / (self.n_features * (self.n_features + 1))
        return np.vstack((jac_f1, jac_f2, jac_f3))


class ZDT1(Problem):
    r"""n_features = 30 (default), n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = x_1, \\
        f_2(x) = h(x) \left( 1 - \sqrt{\frac{x_1}{h(x)}} \right),
        \end{gathered}

    where

    .. math::

        h(x) = 1 + \frac{9}{n - 1} \sum_{i=2}^n x_i.

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (1, 0, \dots, 0)^\top, \\
        \nabla f_2(x) = (- \frac{\sqrt{h(x) / x_1}}{2},
            \frac{9}{2 (n - 1)} (1 - \sqrt{x_1 / h(x)}),
            \dots, \frac{9}{2 (n - 1)} (1 - \sqrt{x_1 / h(x)}) )^\top.
        \end{gathered}

    Reference: Zitzler, E., Deb, K., Thiele, L.: Comparison of multiobjective
    evolutionary algorithms: empirical results. Evolutionary Computation,
    IEEE Transactions on 8(2), 257–271 (2000)
    """

    def __init__(self, n_features: int = 30) -> None:
        super().__init__(n_features=n_features, n_objectives=2, bounds=(1e-6, np.inf))

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = x[0]
        h = 1 + 9 / (self.n_features - 1) * np.sum(x[1:])
        f2 = h * (1 - np.sqrt(f1 / h))
        return np.array([f1, f2])

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = np.zeros(self.n_features)
        jac_f1[0] = 1
        h = 1 + 9 / (self.n_features - 1) * np.sum(x[1:])
        jac_f2 = np.full(
            self.n_features, 9 * (2 - np.sqrt(x[0] / h)) / 2 / (self.n_features - 1)
        )
        jac_f2[0] = -np.sqrt(h / x[0]) / 2
        return np.vstack((jac_f1, jac_f2))


class TOI4(Problem):
    r"""n_features = 4, n_objectives = 2

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = x_1^2 + x_2^2 + 1,
        f_2(x) = 0.5((x_1 - x_2)^2 + (x_3 - x_4)^2) + 1.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (2 x_1, 2 x_2, 0, 0)^\top,
        \nabla f_2(x) = (x_1 - x_2, x_2 - x_1, x_3 - x_4, x_4 - x_3)^\top.
        \end{gathered}

    Reference: Toint, Ph.L.: Test problems for partially separable optimization
    and results for the routine PSPMIN. Tech. Rep. 83/4, Department of Mathematics,
    University of Namur, Brussels (1983)
    """

    def __init__(
        self,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[NDArray[np.floating[T]] | float, NDArray[np.floating[T]] | float]
        | None = None,
    ) -> None:
        super().__init__(
            n_features=4,
            n_objectives=2,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        f1 = x[0] ** 2 + x[1] ** 2 + 1
        f2 = 0.5 * ((x[0] - x[1]) ** 2 + (x[2] - x[3]) ** 2) + 1
        return np.array([f1, f2])

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        jac_f1 = np.zeros(self.n_features)
        jac_f1[0] = 2 * x[0]
        jac_f1[1] = 2 * x[1]
        jac_f2 = np.zeros(self.n_features)
        jac_f2[0] = x[0] - x[1]
        jac_f2[1] = -jac_f2[0]
        jac_f2[2] = x[2] - x[3]
        jac_f2[3] = -jac_f2[2]
        return np.vstack((jac_f1, jac_f2))


class TRIDIA(Problem):
    r"""n_features = 3, n_objectives = 3

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_1(x) = (2 x_1 - 1)^2,
        f_2(x) = 2 (2 x_1 - x_2)^2,
        f_3(x) = 3 (2 x_2 - x_3)^2.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_1(x) = (8 x_1 - 4, 0, 0)^\top,
        \nabla f_2(x) = (16 x_1 - 8 x_2, 4 x_2 - 8 x_1, 0)^\top,
        \nabla f_3(x) = (0, 24 x_2 - 12 x_3, 6 x_3 - 12 x_2)^\top.
        \end{gathered}

    Reference: Toint, Ph.L.: Test problems for partially separable optimization and
    results for the routine PSPMIN. Tech. Rep. 83/4, Department of Mathematics,
    University of Namur, Brussels (1983)
    """

    def __init__(
        self,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[NDArray[np.floating[T]] | float, NDArray[np.floating[T]] | float]
        | None = None,
    ) -> None:
        super().__init__(
            n_features=3,
            n_objectives=3,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return np.array(
            [
                (2 * x[0] - 1) ** 2,
                2 * (2 * x[0] - x[1]) ** 2,
                3 * (2 * x[1] - x[2]) ** 2,
            ]
        )

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return np.array(
            [
                [8 * x[0] - 4, 0, 0],
                [16 * x[0] - 8 * x[1], 4 * x[1] - 8 * x[0], 0],
                [0, 24 * x[1] - 12 * x[2], 6 * x[2] - 12 * x[1]],
            ]
        )


class LinearFunctionRank1(Problem):
    r"""n_features = 10 (default), n_objectives = 4 (default)

    We solve problems with the objective functions

    .. math::

        \begin{gathered}
        f_i(x) = \left( i \sum_{j = 1}^n j x_j - 1 \right)^2, \quad i = 1, \dots, 4.
        \end{gathered}

    Each gradient of :math:`f_i` can be written as

    .. math::

        \begin{gathered}
        \nabla f_i(x) = \left[ 2 i k \left( i \sum_{j = 1}^n j x_j - 1 \right) \right]_k
        \end{gathered}

    Reference: Moré, J.J., Garbow, B.S., Hillstrom, K.E.: Testing unconstrained
    optimization software. ACM T. Math. Softw. 7(1), 17–41 (1981)
    """

    def __init__(
        self,
        n_features: int = 10,
        n_objectives: int = 4,
        l1_ratios: Sequence[float] | None = None,
        l1_shifts: Sequence[float] | None = None,
        bounds: tuple[FloatArray | float, FloatArray | float] | None = None,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_objectives=n_objectives,
            l1_ratios=l1_ratios,
            l1_shifts=l1_shifts,
            bounds=bounds,
        )
        self.range_n_objectives = np.arange(1, self.n_objectives + 1)
        self.range_n_features = np.arange(1, self.n_features + 1)

    def f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return cast(
            NDArray[np.floating[T]],
            (self.range_n_objectives * np.inner(self.range_n_features, x) - 1) ** 2,
        )

    def jac_f(self, x: NDArray[np.floating[T]]) -> NDArray[np.floating[T]]:
        if self.n_features != len(x):
            raise ValueError(f"len(x) should be equal to n_features, got {x}.")
        return cast(
            NDArray[np.floating[T]],
            2
            * self.range_n_objectives[:, None]
            * self.range_n_features
            * (
                self.range_n_objectives[:, None] * np.inner(self.range_n_features, x)
                - 1
            ),
        )
