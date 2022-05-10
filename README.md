# zfista : A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective convex optimization

This code repository includes the source code for the [Paper](http://example.com "Preparing")

```
A globally convergent fast iterative shrinkage-thresholding algorithm with a new momentum factor for single and multi-objective convex optimization
Hiroki Tanabe, Ellen H. Fukuda, and Nobuo Yamashita
```

## Requirements
- Python 3.5 or later

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
```sh
jupyter notebook
```

## For developers
To set up development environment, run
```sh
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Testing
You can run all tests by
```sh
python -m unittest discover
```
