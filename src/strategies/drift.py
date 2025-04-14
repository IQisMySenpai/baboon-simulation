"""
Here we define the drift and diffusion functions for the baboon simulation.
"""
import numpy as np
import numpy.typing as npt
from simulation_types.documentation import DriftType  # , DiffusionType
from utils.baboons import (
    get_angles,
    get_differences,
)

# We do not need to specify that we are implementing a function of type
# DriftType or DiffusionType, we just have to implement the function with the
# right signature.


def only_angles_drift_function(
    angle_std: float,
    step_length: float,
) -> DriftType:
    """The drift function that only considers the angles of the baboons.
    The drift has the angle of another baboon plus a normal perturbation
    of standard deviation angle_std. The drift has length step_length.

    Args:
        angle_std: Standard deviation of the normal perturbation of the angle
            after choosing one direction.
        step_length: Length of the drift vectors.
    """
    return lambda baboons_trajectory, rng: step_length * _only_angles_drift(
        baboons_trajectory=baboons_trajectory,
        rng=rng,
        angle_std=angle_std,
    )


def _only_angles_drift(
    baboons_trajectory: npt.NDArray[np.float64],
    rng: np.random.Generator,
    angle_std: float,
) -> npt.NDArray[np.float64]:
    """
    Drift function that randomly selects the angle of another baboon plus a
    normal perturbation.
    
    This makes the angle distribution smooth, because the resulting
    distribution is the convolution of the normal perturbation with the
    discrete distribution that assigns probability 1/(n_baboons - 1) to the
    angle wrt each other baboon.

    Args:
        baboons_trajectory: Full trajectory of baboons. Shape (t, n_baboons, 2)
        rng: Random generator
        std: Standard deviation of the normal perturbation

    Returns:
        Drift vector of length 1 for each baboon. Shape (n_baboons, 2)
    """
    n_baboons = baboons_trajectory.shape[1]
    current_baboons = baboons_trajectory[-1, :, :]
    baboons_angles = get_angles(current_baboons)
    # select a random baboon's angle but ensure that it is not itself
    random_indices = rng.integers(0, n_baboons - 1, size=n_baboons)
    random_indices[random_indices >= np.arange(n_baboons)] += 1

    chosen_angles = baboons_angles[np.arange(n_baboons), random_indices]
    chosen_angles_perturbed = (
        chosen_angles + rng.normal(0, angle_std, size=n_baboons)
    )
    drift = np.column_stack(
        (
            np.cos(chosen_angles_perturbed),
            np.sin(chosen_angles_perturbed),
        )
    )
    return drift
