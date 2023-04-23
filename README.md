# zfista : A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective (convex) optimization

[![Actions](https://github.com/zalgo3/zfista/actions/workflows/actions.yaml/badge.svg)](https://github.com/zalgo3/zfista/actions/workflows/actions.yaml)
[![PyPI version](https://badge.fury.io/py/zfista.svg)](https://badge.fury.io/py/zfista)

This code repository provides a solver for the proximal gradient method (ISTA) and its acceleration (FISTA) for both single and multi-objective optimization problems, including the experimental code for the [Paper1](https://arxiv.org/abs/2202.10994 "An accelerated proximal gradient method for multiobjective optimization") and [Paper2](https://arxiv.org/abs/2205.05262 "A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective convex optimization").

```txt:Paper1
An accelerated proximal gradient method for multiobjective optimization
Hiroki Tanabe, Ellen H. Fukuda, and Nobuo Yamashita
```

```txt:Paper2
A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective convex optimization
Hiroki Tanabe, Ellen H. Fukuda, and Nobuo Yamashita
```

The solver can deal with the unconstrained problem written by
$$\min_{x \in \mathbf{R}^n} \quad F(x) \coloneqq f(x) + g(x),$$
where $f$ and $g$ are scalar or vector valued function, $f$ is continuously differentiable, $g$ is closed, proper and convex.
Note that FISTA also requires $f$ to be convex.

- Documentation: https://zalgo3.github.io/zfista/

## Requirements
- Python 3.8 or later

## Install
```sh
pip install zfista
```

## Quickstart
```python
from zfista import minimize_proximal_gradient
help(minimize_proximal_gradient)
```

## Examples
You can run some examples on jupyter notebooks.
```Shell
jupyter notebook
```

## Testing
You can run all tests by
```Shell
python -m unittest discover
```

## Benchmark
You can run the benchmark by
```Shell
python runtests.py
```
