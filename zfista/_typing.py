from __future__ import annotations

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing import Union

    from typing_extensions import TypeAlias

FloatArray: TypeAlias = NDArray[np.floating[Any]]

if sys.version_info >= (3, 10):
    Scalar: TypeAlias = np.floating[Any] | float
else:
    Scalar: TypeAlias = Union[np.floating[Any], float]
