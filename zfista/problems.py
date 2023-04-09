import numpy as np
from scipy.optimize import root_scalar


def _soft_threshold(x, thresh):
    return np.where(np.abs(x) <= thresh, 0, x - thresh * np.sign(x))


class Problem:
    """Superclass of test problems to be solved by the proximal gradient methods for
    multiobjective optimization.

    In all test problems, each objective function can be written as

        F_i(x) = f_i(x) + g_i(x),

    where f_i is convex and differentiable and g_i is closed, proper and convex.

    Parameters
    ----------
    n_dims : int
        The dimension of the decision variable.

    m_dims : int
        The number of objective functions.
    """

    def __init__(self, n_dims, m_dims):
        self.n_dims = n_dims
        self.m_dims = m_dims

    def g(self, x):
        return np.zeros(self.m_dims)

    def prox_wsum_g(self, weight, x):
        return x


class JOS1(Problem):
    """n_dims = 5 (default), m_dims = 2

    We solve problems with the objective functions::

        f_1(x) = (1 / n_dims) * \\sum_i x_i^2,          g_1(x) = 0,
        f_2(x) = (1 / n_dims) * \\sum_i (x_i - 2)^2,    g_2(x) = 0.

    Each gradient of f_i can be written as::

        \\nabla f_1(x) = (2 / n_dims) * x,
        \\nabla f_2(x) = (2 / n_dims) * (x - 2).

    Reference: Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for evolutionary
    multi-objective optimization: Why does it work and how? In: GECCO’01 Proceedings of the 3rd
    Annual Conference on Genetic and Evolutionary Computation, pp. 1042–1049 (2001)
    """

    def __init__(self, n_dims=5, m_dims=2):
        super().__init__(n_dims=n_dims, m_dims=m_dims)

    def f(self, x):
        f1 = np.linalg.norm(x) ** 2 / self.n_dims
        f2 = np.linalg.norm(x - 2) ** 2 / self.n_dims
        return np.array([f1, f2])

    def jac_f(self, x):
        jac_f1 = 2 * x / self.n_dims
        jac_f2 = 2 * (x - 2) / self.n_dims
        return np.vstack((jac_f1, jac_f2))


class JOS1_L1(JOS1):
    """n_dims = 5 (default), m_dims = 2

    We solve the modified version of `JOS1`, where::

        g_1(x) = r_1 * ||x||_1, g_2(x) = r_2 * ||x - 1||_1.

    The proximal operator of the weighted sum of g_i can be written as::

        prox_{\\sum_i w_i g_i}(x) = S_{r_1 w_1}(S_{r_0 w_0}(x + r_1 w_1) - r_1 w_1 - 1) + 1,

    where S is the soft-thresholding operator.
    """

    def __init__(self, n_dims=5, m_dims=2, l1_ratios=None):
        super().__init__(n_dims=n_dims, m_dims=m_dims)
        if l1_ratios is None:
            l1_ratios = (1 / n_dims, 1 / 2 / n_dims)
        self.l1_ratios = l1_ratios

    def g(self, x):
        g1 = np.linalg.norm(x, ord=1) * self.l1_ratios[0]
        g2 = np.linalg.norm(x - 1, ord=1) * self.l1_ratios[1]
        return np.array([g1, g2])

    def prox_wsum_g(self, weight, x):
        return (
            _soft_threshold(
                _soft_threshold(
                    x + weight[1] * self.l1_ratios[1], weight[0] * self.l1_ratios[0]
                )
                - weight[1] * self.l1_ratios[1]
                - 1,
                weight[1] * self.l1_ratios[1],
            )
            + 1
        )


class SD(Problem):
    """n_dims = 4, m_dims = 2

    We solve problems with the objective functions::

        f_1(x) = 2 x_1 + \\sqrt{2} x_2 + \\sqrt{2} x_3 + x_4, g_1(x) = ind([lb, ub]),
        f_2(x) = 0, g_2(x) = 2 / x_1 + 2 \\sqrt{2} / x_2 + 2 \\sqrt{2} / x_3 + x_4 + ind([lb, ub]),

    where ind represents the indicator function, and [lb, ub] is upper and lower bound::

        lb = [1, \\sqrt{2}, \\sqrt{2}, 1], ub = [3, 3, 3, 3].

    Each gradient of f_i can be written as::

        \\nabla f_1(x) = [1, \\sqrt{2}, \\sqrt{2}, 1], \\nabla f_2(x) = 0.

    Since g_i is separable in the effective domain, the proximal operator of the weighted sum of
    g_i can be computed by the weighted sum of the proximal operator of each g_i.

    Reference: Stadler, W., Dauer, J.: Multicriteria optimization in engineering: a tutorial and
    survey. In: Kamat, M.P. (ed.) Progress in Aeronautics and Astronautics: Structural
    Optimization: Status and Promise, vol. 150, pp. 209–249. American Institute of Aeronautics
    and Astronautics, Reston (1992)
    """

    def __init__(
        self,
        n_dims=4,
        m_dims=2,
        lb=np.array([1, np.sqrt(2), np.sqrt(2), 1]),
        ub=np.array([3, 3, 3, 3]),
    ):
        super().__init__(n_dims=n_dims, m_dims=m_dims)
        self.lb = lb
        self.ub = ub

    def f(self, x):
        f1 = 2 * x[0] + np.sqrt(2) * x[1] + np.sqrt(2) * x[2] + x[3]
        f2 = 0
        return np.array([f1, f2])

    def jac_f(self, x):
        jac_f1 = np.array([2, np.sqrt(2), np.sqrt(2), 1])
        jac_f2 = np.zeros_like(jac_f1)
        return np.vstack((jac_f1, jac_f2))

    def g(self, x):
        if np.any(x < self.lb) or np.any(x > self.ub):
            g1 = g2 = np.inf
        else:
            g1 = 0
            g2 = 2 / x[0] + 2 * np.sqrt(2) / x[1] + 2 * np.sqrt(2) / x[2] + 2 / x[3]
        return np.array([g1, g2])

    def prox_wsum_g(self, weight, x):
        ret = np.empty(self.n_dims)
        constants = np.array([2, 2 * np.sqrt(2), 2 * np.sqrt(2), 2])
        for i in range(self.n_dims):
            ret[i] = root_scalar(
                lambda z: z**3 - x[i] * z**2 - constants[i] * weight[1],
                x0=x[i],
                fprime=lambda z: 3 * z**2 - 2 * x[i] * z,
                fprime2=lambda z: 6 * z - 2 * x[i],
            ).root
        ret = np.where(ret < self.lb, self.lb, ret)
        ret = np.where(ret > self.ub, self.ub, ret)
        return ret


class FDS(Problem):
    """n_dims = 10 (default), m_dims = 3

    We solve problems with the objective functions::

        f_1(x) = \\sum_i i (x_i - i)^4 / n_dims^2,                               g_1(x) = 0,
        f_2(x) = exp(\\sum_i x_i / n_dims) + ||x||^2,                            g_2(x) = 0,
        f_3(x) = \\sum_i i (n_dims - i + 1) exp(-x_i) / (n_dims * (n_dims + 1)), g_3(x) = 0.

    Each gradient of f_i can be written as::

        \\nabla f_1(x) = 4 / n_dims^2 * \\sum_i i (x_i - i)^3,
        \\nabla f_2(x) = exp(\\sum_i x_i / n_dims) / n_dims + 2 * x,
        \\nabla f_3(x) = - [i (n_dims - i + 1) exp(-x_i) / (n_dims * (n_dims + 1))]_i

    Reference: Fliege, J., Graña Drummond, L.M., Svaiter, B.F.: Newton’s method for multiobjective
    optimization. SIAM J. Optim. 20(2), 602–626 (2009)
    """

    def __init__(self, n_dims=10, m_dims=3):
        super().__init__(n_dims=n_dims, m_dims=m_dims)
        self.one_to_n = np.arange(self.n_dims) + 1
        self.conv_n = self.one_to_n * self.one_to_n[::-1]

    def f(self, x):
        f1 = np.inner(self.one_to_n, (x - self.one_to_n) ** 4) / self.n_dims**2
        f2 = np.exp(x.sum() / self.n_dims) + np.linalg.norm(x) ** 2
        f3 = np.inner(self.conv_n, np.exp(-x)) / (self.n_dims * (self.n_dims + 1))
        return np.array([f1, f2, f3])

    def jac_f(self, x):
        jac_f1 = 4 / self.n_dims**2 * self.one_to_n * (x - self.one_to_n) ** 3
        jac_f2 = np.exp(x.sum() / self.n_dims) / self.n_dims + 2 * x
        jac_f3 = -self.conv_n * np.exp(-x) / (self.n_dims * (self.n_dims + 1))
        return np.vstack((jac_f1, jac_f2, jac_f3))


class FDS_CONSTRAINED(FDS):
    """n_dims = 10 (default), m_dims = 3

    We solve the modified version of `FDS`, where::

        g_1(x) = g_2(x) = g_3(x) is the indicator function of the nonnegative orthant.

    The proximal operator of the weighted sum of g_i can be written as::

        prox_{\\sum_i w_i g_i}(x) = max(x, -2).
    """

    def __init__(self, n_dims=10, m_dims=3):
        super().__init__(n_dims=n_dims, m_dims=m_dims)

    def g(self, x):
        if np.any(x < 0):
            g1 = g2 = g3 = np.inf
        else:
            g1 = g2 = g3 = 0
        return np.array([g1, g2, g3])

    def prox_wsum_g(self, weight, x):
        return np.maximum(x, 0)
