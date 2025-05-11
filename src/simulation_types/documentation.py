"""
We are going to use weak (non-interpreter-enforced) protocols to define
objects. Further down the line we could eventually use
https://peps.python.org/pep-0544/, but objects right now are too simple to
require other machinery such as classes.
Refer to https://peps.python.org/pep-0020/:
    "Simple is better than complex."

Here is the documentation for said objects.


# Objects

baboons: np.ndarray: (M, 2)-np.ndarray (i.e. Mx2 matrix)
    - M is the number of baboons.
    - 2 is the x and y coordinates of the baboon.

baboons_trajectory: (N, M, 2)-np.ndarray (i.e. NxMx2 matrix)
    - N is the number of steps.
    - M is the number of baboons.
    - 2 is the x and y coordinates of the baboon.
For example baboons_trajectory[t, i, :] is the position of baboon i at time t.

colors: List[str]: List of colors for each baboon. (List of length M)


# SDE Model

The i-th baboon position is given by the following Stochastic Differential
Equation (SDE):

dx_i = f(x[:t], omega) dt + g(x[:t], omega) dW_t(omega),

where:
    - dx_i is the change in position of baboon i.
    - x[:t] is the full trajectory of all baboons up to time t.
        It is a (t, M, 2)-np.ndarray.
    - f(x[:t], omega) is the DRIFT of the SDE.
        It denotes the average change in position of baboon i.
        This term may include a random component (thus the dependency on
        omega).
        f outputs an (M, 2)-np.ndarray.
    - g(x[:t], omega) is the DIFFUSION of the SDE.
        It denotes the random change in position of baboon i.
        g outputs an (M, 2, J)-np.ndarray.
    - W_t(omega) is a (MxJ-dimensional)-Wiener process (Brownian motion).

In practice, this will be implemented with an Euler scheme:
    x[t+1] = x[t] + f(x[:t]) * dt + g(x[:t]) * dW_t,
where dt is the fixed time step size and dW_t is a normal random variable
with mean 0 and variance dt. Multiplication "*" here is supposed to be in a
matrix sense:
    g(x[:t]) * dW_t = np.einsum("mij,mj->mi", g(x[:t]), dW_t).

We will firstly consider cases with g(x[:t]) = 0, i.e. no diffusion.
If we add the noise term later, we will have to decide what the dimension
J of the Wiener process is.

We will use the following notation in the code:
    - f == drift
    - g == diffusion
    - x[:t] == baboons_trajectory[:t]

Note: we could be more general, were the output of the diffusion function is
(M, 2, J, 2) and the BM is (J, 2), and their multiplication is done
with Einstein summation convention: mijk, jk -> mi.
"""
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from sklearn.utils import Bunch

# The new way to use drift and diffusion: with a state.
# See implementation in strategies/drift_with_state.py
DriftDiffusionWithStateType = Callable[
    [
        # baboons_trajectory[:t], shape (t, n_baboons, 2)
        npt.NDArray[np.float64],
        # random generator (this is the omega)
        np.random.Generator,
        # state
        Optional[Bunch],
    ],
    # (output of drift, output of diffusion, next_state)
    tuple[
        npt.NDArray[np.float64],  # shape (n_baboons, 2)
        npt.NDArray[np.float64],  # shape (n_baboons, 2, J)
        Bunch,
    ],
]


# Signature for the drift and diffusion functions
DriftType = Callable[
    [
        # baboons_trajectory[:t], shape (t, n_baboons, 2)
        npt.NDArray[np.float64],
        # random generator (this is the omega)
        np.random.Generator,
        # state
        Bunch,
    ],
    # (output of drift, next_state)
    tuple[
        npt.NDArray[np.float64],  # (n_baboons, 2)
        Bunch,
    ],
]

DiffusionType = Callable[
    [
        # baboons_trajectory[:t], shape (t, n_baboons, 2)
        npt.NDArray[np.float64],
        # random generator (this is the omega)
        np.random.Generator,
        # state
        Bunch,
    ],
    # output of diffusion, shape (n_baboons, J)
    npt.NDArray[np.float64],
]
