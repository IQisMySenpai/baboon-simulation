import numpy as np
from enum import Enum
from sklearn.utils import Bunch
import numpy.typing as npt
from typing import Callable, Optional, Tuple
from simulation_types.documentation import DriftDiffusionWithStateType
from utils.baboons import get_angles, get_differences


class State(Enum):
    """State of the baboon.
    The baboon can be in one of the following states:
        - following: The baboon is following another baboon.
        - group_influence: The baboon is influenced by the group (choose a
            random angle pointing to another).
        - still: The baboon is not moving (maybe only moving with a small
            perturbation).
        - random_walk: The baboon is doing a random walk (i.e. exploring on its
            own).
    """
    following = 1
    group_influence = 2
    still = 3
    random_walk = 4


def state_driven_drift_diffusion_function(
    angle_std: float,
    step_length: float,
    max_follow_step: float,
    countdown_poisson_mean: float = 20.0,
    sigma_still: float = 0.01,
    sigma_following: float = 0.1,
    sigma_group_influence: float = 0.1,
    sigma_random_walk: float = 0.5,
    following_step_size_std: float = 0.05,
    following_step_size_proportion: float = 0.1,
    state_probabilities: Optional[dict[State, float]] = None,
    state_countdown_means: Optional[dict[State, float]] = None,
) -> DriftDiffusionWithStateType:
    """
    Creates a drift + diffusion function where each baboon acts according to
    an internal state.
    
    Each baboon can be in one of four states: following, group_influence,
    still, or random_walk.
    State transitions occur after a countdown, drawn from a Poisson
    distribution.

    Args:
        angle_std (float): Standard deviation for angular perturbations in
            group_influence state.
        step_length (float): Base step length for group_influence and random
            walk drift.
        max_follow_step (float): Maximum step size a baboon can take while
            following another baboon.
        countdown_poisson_mean (float): Default mean of the Poisson distribution
            used to sample state countdowns if per-state means are not provided.
        sigma_still (float): Diffusion coefficient for still baboons.
        sigma_following (float): Diffusion coefficient for following baboons.
        sigma_group_influence (float): Diffusion coefficient for baboons under
            group influence.
        sigma_random_walk (float): Diffusion coefficient for random walking
            baboons.
        following_step_size_std (float): Standard deviation of noise added to
            following step size.
        following_step_size_proportion (float): Proportion of the distance to
            the target used as a base for following step size.
        state_probabilities (Optional[dict[State, float]]): 
            A dictionary mapping each State to its probability when sampling
            new states. If None, defaults to equal probability for all states.
        state_countdown_means (Optional[dict[State, float]]):
            Optional dictionary specifying Poisson means for each state.
            If a state's mean is not specified, the default
            `countdown_poisson_mean` is used.

    Returns:
        DriftDiffusionWithStateType: A callable that computes the drift vector,
        diffusion matrix, and updated internal state for all baboons given
        their trajectory history and a random generator.

    Notes:
        - In random_walk state, baboons move with a persistent random drift
            direction plus diffusion.
        - The internal state includes an additional field `random_walk_drift`
            to store assigned random walk drifts.
    """

    if state_probabilities is None:
        state_probabilities = {
            State.following: 0.25,
            State.group_influence: 0.25,
            State.still: 0.25,
            State.random_walk: 0.25,
        }
    state_list = list(state_probabilities.keys())
    prob_list = list(state_probabilities.values())

    def drift_diffusion(
        baboons_trajectory: npt.NDArray[np.float64],
        rng: np.random.Generator,
        state_bunch: Optional[Bunch],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Bunch]:
        n_baboons = baboons_trajectory.shape[1]
        current_positions = baboons_trajectory[-1, :, :]  # (n_baboons, 2)

        if state_bunch is None:
            state_bunch = Bunch(
                state=np.full(n_baboons, State.random_walk.value),
                following_idx=np.arange(n_baboons),
                state_countdown=np.zeros(n_baboons, dtype=int),
                random_walk_drift=np.zeros((n_baboons, 2)),
            )

        next_state = Bunch(
            state=state_bunch.state.copy(),
            following_idx=state_bunch.following_idx.copy(),
            state_countdown=state_bunch.state_countdown.copy(),
            random_walk_drift=state_bunch.random_walk_drift.copy(),
        )

        # Update states based on countdown
        countdown_zero = (next_state.state_countdown <= 0)
        if np.any(countdown_zero):
            n_updates = np.sum(countdown_zero)
            # Sample new states
            new_states = rng.choice(
                [s.value for s in state_list], size=n_updates, p=prob_list,
            )
            next_state.state[countdown_zero] = new_states

            # Sample new following targets
            next_state.following_idx[countdown_zero] = rng.integers(
                0, n_baboons, size=n_updates,
            )

            # Sample new countdowns
            countdowns = np.empty(n_updates, dtype=int)
            for i, state_value in enumerate(new_states):
                state_enum = State(state_value)
                mean = (
                    countdown_poisson_mean
                    if state_countdown_means is None
                    else state_countdown_means.get(state_enum, countdown_poisson_mean)
                )
                countdowns[i] = 1 + rng.poisson(mean)
            next_state.state_countdown[countdown_zero] = countdowns

            # For new random_walk baboons, assign a random drift
            is_new_random_walk = (new_states == State.random_walk.value)
            if np.any(is_new_random_walk):
                idx_random_walk = (
                    np.flatnonzero(countdown_zero)[is_new_random_walk]
                )
                random_angles = rng.uniform(
                    0, 2 * np.pi, size=idx_random_walk.shape[0],
                )
                random_step_lengths = step_length + rng.normal(
                    0, following_step_size_std, size=idx_random_walk.shape[0],
                )
                random_drifts = np.column_stack(
                    (np.cos(random_angles), np.sin(random_angles))
                ) * random_step_lengths[:, np.newaxis]
                next_state.random_walk_drift[idx_random_walk] = random_drifts

        # Decrement countdowns
        next_state.state_countdown -= 1

        # Initialize drift and diffusion
        drift_vectors = np.zeros_like(current_positions)
        diffusion_matrices = np.zeros((n_baboons, 2, 2))

        # ========== FOLLOWING ==========
        is_following = next_state.state == State.following.value
        if np.any(is_following):
            idx_following = np.flatnonzero(is_following)
            targets = next_state.following_idx[is_following]

            differences = (
                current_positions[targets] - current_positions[idx_following]
            )
            distances = np.linalg.norm(differences, axis=1, keepdims=True)
            directions = differences / np.maximum(distances, 1e-8)

            step_sizes = np.clip(
                (
                    following_step_size_proportion * distances.squeeze()
                    + rng.normal(
                        0, following_step_size_std, size=distances.shape[0],
                    )
                ),
                0.0,
                max_follow_step,
            )
            drift_vectors[idx_following] = (
                directions * step_sizes[:, np.newaxis]
            )

            diffusion_matrices[is_following, 0, 0] = sigma_following
            diffusion_matrices[is_following, 1, 1] = sigma_following

        # ========== GROUP INFLUENCE ==========
        is_group_influence = next_state.state == State.group_influence.value
        if np.any(is_group_influence):
            idx_self = np.flatnonzero(is_group_influence)
            angles = get_angles(current_positions)

            random_choices = rng.integers(
                0, n_baboons - 1, size=idx_self.shape[0],
            )
            random_choices += (random_choices >= idx_self)
            chosen_angles = angles[idx_self, random_choices]

            perturbed_angles = chosen_angles + rng.normal(
                0, angle_std, size=idx_self.shape[0],
            )
            drift_vectors[idx_self] = step_length * np.column_stack(
                (np.cos(perturbed_angles), np.sin(perturbed_angles))
            )

            diffusion_matrices[is_group_influence, 0, 0] = \
                diffusion_matrices[is_group_influence, 1, 1] = \
                sigma_group_influence

        # ========== STILL ==========
        is_still = next_state.state == State.still.value
        if np.any(is_still):
            diffusion_matrices[is_still, 0, 0] = sigma_still
            diffusion_matrices[is_still, 1, 1] = sigma_still

        # ========== RANDOM WALK ==========
        is_random_walk = next_state.state == State.random_walk.value
        if np.any(is_random_walk):
            drift_vectors[is_random_walk] = (
                next_state.random_walk_drift[is_random_walk]
            )
            diffusion_matrices[is_random_walk, 0, 0] = sigma_random_walk
            diffusion_matrices[is_random_walk, 1, 1] = sigma_random_walk

        return drift_vectors, diffusion_matrices, next_state

    return drift_diffusion
