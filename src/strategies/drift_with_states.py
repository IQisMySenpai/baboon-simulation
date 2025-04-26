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
    group_influence_step_length: float,
    random_walk_step_length: float,
    min_follow_distance: float,
    min_follow_step: float,
    max_follow_step: float,
    state_diffusion_constants: dict[State, float],
    following_step_size_std: float = 0.2,
    following_step_size_proportion: float = 0.1,
    state_probabilities: Optional[dict[State, float]] = None,
    state_countdown_means: Optional[dict[State, float]] = None,
) -> DriftDiffusionWithStateType:
    """
    Creates a drift + diffusion function where each baboon acts according to
    an internal state.

    Each baboon can be in one of four states: following, group_influence,
    still, or random_walk. State transitions occur after a countdown, drawn
    from a Poisson distribution.

    Args:
        angle_std (float): Standard deviation for angular perturbations in
            group_influence state.
        group_influence_step_length (float): Base step length for
            group_influence.
        random_walk_step_length (float): base step length for random walk
            drift.
        min_follow_distance (float): Minimum distance between two baboons for
            one to follow the other.
        min_follow_step (float): Minimum step size a baboon can take while
            following another baboon.
        max_follow_step (float): Maximum step size a baboon can take while
            following another baboon.
        state_diffusion_constants (dict[State, float]): Diffusion coefficient
            for each state.
        following_step_size_std (float): Standard deviation of noise added to
            following step size.
        following_step_size_proportion (float): Proportion of the distance to
            the target used as a base for following step size.
        state_probabilities (Optional[dict[State, float]]):
            A dictionary mapping each State to its probability when sampling
            new states. If None, defaults to equal probability for all states.
        state_countdown_means (Optional[dict[State, float]]):
            Dictionary specifying Poisson means for each state.
            Defaults to 20 for all states.

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
    if state_countdown_means is None:
        state_countdown_means = {
            State.following: 20,
            State.group_influence: 20,
            State.still: 20,
            State.random_walk: 20,
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

        # Reset baboons that have reached their countdown
        countdown_zero = (next_state.state_countdown <= 0)
        if np.any(countdown_zero):
            n_updates = np.sum(countdown_zero)
            # Sample new states
            new_states = rng.choice(
                [s.value for s in state_list], size=n_updates, p=prob_list,
            )
            next_state.state[countdown_zero] = new_states

            # Sample new following targets
            idx_updates = np.flatnonzero(countdown_zero)
            for i, idx in enumerate(idx_updates):
                if new_states[i] == State.following.value:
                    distances = np.linalg.norm(
                        current_positions - current_positions[idx], axis=1,
                    )
                    # Only choose target baboons that are not too close
                    possible_targets = np.flatnonzero(
                        distances > min_follow_distance,
                    )
                    if possible_targets.size > 0:
                        next_state.following_idx[idx] = rng.choice(
                            possible_targets,
                        )
                    else:  # If no valid targets, assign random walk state
                        next_state.state[idx] = State.random_walk.value
                        next_state.following_idx[idx] = 0
                else:
                    next_state.following_idx[idx] = 0

            # Sample new countdowns
            countdowns = np.empty(n_updates, dtype=int)
            for i, state_value in enumerate(new_states):
                state_enum = State(state_value)
                mean = state_countdown_means[state_enum]
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
                random_step_lengths = random_walk_step_length + rng.normal(
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
                min_follow_step,
                max_follow_step,
            )
            drift_vectors[idx_following] = (
                directions * step_sizes[:, np.newaxis]
            )

            diffusion_matrices[is_following, :, :] = (
                np.eye(2) * state_diffusion_constants[State.following]
            )

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
            drift_vectors[idx_self] = group_influence_step_length * (
                np.column_stack(
                    (np.cos(perturbed_angles), np.sin(perturbed_angles))
                )
            )

            diffusion_matrices[is_group_influence, :, :] = (
                np.eye(2) * state_diffusion_constants[State.group_influence]
            )

        # ========== STILL ==========
        is_still = next_state.state == State.still.value
        if np.any(is_still):
            diffusion_matrices[is_still, :, :] = (
                np.eye(2) * state_diffusion_constants[State.still]
            )

        # ========== RANDOM WALK ==========
        is_random_walk = next_state.state == State.random_walk.value
        if np.any(is_random_walk):
            drift_vectors[is_random_walk] = (
                next_state.random_walk_drift[is_random_walk]
            )
            diffusion_matrices[is_random_walk, :, :] = (
                np.eye(2) * state_diffusion_constants[State.random_walk]
            )

        return drift_vectors, diffusion_matrices, next_state

    return drift_diffusion
