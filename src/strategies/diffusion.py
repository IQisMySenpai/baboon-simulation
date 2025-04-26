"""
Here we define the diffusion functions for the baboon simulation.
"""
import numpy as np
import numpy.typing as npt
from simulation_types.documentation import DiffusionType
from utils.baboons import (
    get_angles,
    get_differences,
)


def constant_diffusion_function(
    constant_drift: float,
) -> DiffusionType:
    """Constant drift function.

    Args:
        constant_drift: Constant drift for the baboons. Each coordinate of the
            Brownian Motion is multiplied by this constant. There is one
            independent Brownian motion for each coordinate of each baboon.
    """
    return lambda baboons_trajectory, rng: constant_drift * np.eye(
        baboons_trajectory.shape[1],
    )
