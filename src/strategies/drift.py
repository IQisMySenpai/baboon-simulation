"""
Here we define the drift functions for the baboon simulation.
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
        std: Standard deviation of the normal perturbation (could be 0 if no
            perturbation is needed)

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


def alvaros_drift_function(
    angle_std: float,
    step_length: float,
) -> DriftType:
    """I dont know yet what it'll do
    """
    return lambda baboons_trajectory, rng: step_length * alvaros_drift(
        baboons_trajectory=baboons_trajectory,
        rng=rng,
        angle_std=angle_std,
    )


def alvaros_drift(
    baboons_trajectory: npt.NDArray[np.float64],
    rng: np.random.Generator,
    angle_std: float,
) -> npt.NDArray[np.float64]:
    """
    Drift function that ... does a bunch of stuff
    Each baboon either: 
        - Follows a random baboon
        - Moves independently towards a target (not yet defined entirely)

    Additional: 
        - Step size adjusted so that they move more towards the closer baboons
        - Baboons that are lost are pulled towards group
        - there is attraction to neighbours


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
    differences = get_differences(current_baboons)
    distances = np.linalg.norm(differences, axis=2)
    #print(distances)

    # Random following (including itself))
    random_indices = rng.integers(0, n_baboons - 1, size=n_baboons)
    chosen_distances = distances[np.arange(n_baboons), random_indices] #distance to the chosen baboon

    # Normalize distances to [0, 1] and reverse scale for weighted step sizes
    min_d = np.min(chosen_distances)
    max_d = np.max(chosen_distances)
    range_d = max_d - min_d if max_d > min_d else 1e-6  # avoid div by zero

    #Config params:
    max_step = 2
    group_threshold = 30.0  
    indep_step = 2  #Extra step size if independent direction
    n_closest = 2
    blend_weight = 0.7 #how much to align with neighbours

    # Outputs
    chosen_angles_perturbed = np.zeros(n_baboons)
    step_magnitudes = np.ones(n_baboons)

    for i in range(n_baboons):
        if random_indices[i] == i: #Independent chosen motion (baboon choosing a direction when it 'follows itself')
            #random_angle = rng.uniform(0, 2 * np.pi)
            #chosen_angles_perturbed[i] = random_angle #random motion
            chosen_angles_perturbed[i] = 0.5*np.pi #move up (biased motion) --> in future towards a target or baboons leaders?
            step_magnitudes[i] = max_step + indep_step #larger step size if it is independent movement

        else: #baboon following other baboon
            # Get angle to the chosen baboon + add noise
            base_angle = baboons_angles[i, random_indices[i]]
            noisy_angle = base_angle + rng.normal(0, angle_std)
            chosen_angles_perturbed[i] = noisy_angle

            # Step size inversely prop to distance 
            normalized = (chosen_distances[i] - min_d) / range_d  # in [0, 1]
            step_magnitudes[i] = max_step * (1 - normalized)

            # Soft attraction toward the closest N baboons
            dists = distances[i].copy()
            dists[i] = np.inf  # exclude self

            # Get indices of closest n baboons
            closest_indices = np.argsort(dists)[:n_closest]
            closest_positions = current_baboons[closest_indices]
            avg_position = np.mean(closest_positions, axis=0)

            vector_to_avg = avg_position - current_baboons[i]
            cohesion_angle = np.arctan2(vector_to_avg[1], vector_to_avg[0])

            # Blend gently with current direction
            chosen_angles_perturbed[i] = (
                (1 - blend_weight) * chosen_angles_perturbed[i]
                + blend_weight * cohesion_angle
            )

        # Group cohesion: join group if too far
        group_center = (np.sum(current_baboons, axis=0) - current_baboons[i]) / (n_baboons - 1)
        vector_to_group = group_center - current_baboons[i]
        distance_to_group = np.linalg.norm(vector_to_group)

        if distance_to_group > group_threshold:
            # Override angle towards group + noise
            chosen_angles_perturbed[i] = np.arctan2(vector_to_group[1], vector_to_group[0])+ rng.normal(0, angle_std)
            #Bonus step size in case too far???????????????????
            extra = np.clip((distance_to_group - group_threshold) / group_threshold, 0, 1)
            step_magnitudes[i] += 10.0 * extra



    # Convert angles to unit vectors
    unit_directions = np.column_stack((
        np.cos(chosen_angles_perturbed),
        np.sin(chosen_angles_perturbed),
    ))
    #distance and individual weighted
    drift = unit_directions * step_magnitudes[:,np.newaxis] #apply individual magnitudes to each row


    return drift

