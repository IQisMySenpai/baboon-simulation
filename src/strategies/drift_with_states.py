import numpy as np
from enum import Enum
from sklearn.utils import Bunch
import numpy.typing as npt
from typing import Optional, Tuple
from simulation_types.documentation import DriftDiffusionWithStateType
from utils.baboons import get_angles, get_distances


class State(Enum):
    """State of the baboon.
    The baboon can be in one of the following states:
        - following: The baboon is following another baboon.
        - group_influence: The baboon is influenced by the group (choose a
            random angle pointing to another).
        - still: The baboon is not moving (maybe only moving with a small
            perturbation).
        - random_walk: The baboon is doing a random walk (i.e. exploring on its
            own) with drift. The drift is randomly assigned at the beginning of
            the random walk and is kept until the baboon changes state.
    """
    following = 1
    group_influence = 2
    still = 3
    random_walk = 4


def state_driven_drift_diffusion_function(
    angle_std: float,
    group_influence_step_length: float,
    random_walk_step_length: float,
    random_walk_step_length_std: float,
    min_follow_distance: float,
    max_follow_distance: float,
    max_follow_step: float,
    state_diffusion_constants: dict[State, float],
    following_step_length_std: float = 0.2,
    following_step_length_proportion: float = 0.1,
    following_radius: float = 1.0,
    state_probabilities: Optional[dict[State, float]] = None,
    state_countdown_means: Optional[dict[State, float]] = None,
    probability_repeat_random_walk: float = 0.0,
    choose_drift_from_other_random_walkers: bool = True,
    new_random_walk_drift_angle_std: float = 10 * np.pi / 180,
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
        random_walk_step_length_std (float): Standard deviation of noise added
            to random walk step size.
        min_follow_distance (float): Minimum distance between two baboons for
            one to follow the other.
        max_follow_distance (float): Maximum distance between two baboons for
            one to follow the other.
        max_follow_step (float): Maximum step size a baboon can take while
            following another baboon.
        following_radius (float): Radius up to which a baboon is satisfied with
            its following target. If the distance to the target is smaller than
            this radius, the baboon will move in the opposite direction.
        state_diffusion_constants (dict[State, float]): Diffusion coefficient
            for each state.
        following_step_length_std (float): Standard deviation of noise added to
            following step size.
        following_step_length_proportion (float): Proportion of the distance to
            the target used as a base for following step size.
        state_probabilities (Optional[dict[State, float]]):
            A dictionary mapping each State to its probability when sampling
            new states. If None, defaults to equal probability for all states.
        state_countdown_means (Optional[dict[State, float]]):
            Dictionary specifying Poisson means for each state.
            Defaults to 20 for all states.
        probability_repeat_random_walk (float): Probability of repeating the
            random walk state when transitioning from random_walk to another
            state.
        choose_drift_from_other_random_walkers (bool): If True, when a baboon
            transitions to the random_walk state, it will choose a drift from
            another baboon that is already in random_walk state plus some angle
            perturbation. If False, the baboon will choose a random drift
            direction.
        new_random_walk_drift_angle_std (float): Standard deviation of the
            angle perturbation for the new random walk drift. This is used when
            a baboon transitions to the random_walk state and is assigned a
            drift based on an existing random walker. The angle is perturbed
            by a normal distribution with this standard deviation.

    Returns:
        DriftDiffusionWithStateType: A callable that computes the drift vector,
        diffusion matrix, and updated internal state for all baboons given
        their trajectory history and a random generator.

    Notes:
        - In random_walk state, baboons move with a persistent random drift
            direction plus diffusion.
        - The internal state includes an additional field `random_walk_drift`
            to store assigned random walk drifts.
        - In the following state, baboons follow a target baboon which has to
            be far enough and in random_walk state.
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
                state=np.full(n_baboons, State.still.value),
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
            idx_updates = np.flatnonzero(countdown_zero)
            # Sample whether to repeat random walk
            repeat_random_walk_flags = (
                rng.uniform(0, 1, size=n_updates) <
                probability_repeat_random_walk
            )
            currently_random_walk = (
                state_bunch.state[idx_updates] == State.random_walk.value
            )
            force_random_walk = (
                repeat_random_walk_flags & currently_random_walk
            )
            new_states[force_random_walk] = State.random_walk.value

            # Sample new following targets
            for i, idx in enumerate(idx_updates):
                if new_states[i] == State.following.value:
                    distances = np.linalg.norm(
                        current_positions - current_positions[idx], axis=1,
                    )
                    # Identify baboons that are both:
                    # - far enough
                    # - currently in random_walk state
                    possible_targets = np.flatnonzero(
                        (distances > min_follow_distance)
                        & (distances < max_follow_distance)
                        & (next_state.state == State.random_walk.value)
                    )

                    if possible_targets.size > 0:
                        target_distances = distances[possible_targets]
                        # Inverse-distance weighting using softmax-like logic
                        weights = np.exp(-target_distances)
                        next_state.following_idx[idx] = rng.choice(
                            possible_targets, p=weights / weights.sum(),
                        )
                    else:  # If no valid targets, assign group influence state
                        next_state.state[idx] = State.group_influence.value
                        new_states[i] = State.group_influence.value
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

            # For new random_walk baboons, assign drift based on existing
            # random_walk baboons
            is_new_random_walk = (new_states == State.random_walk.value)
            if np.any(is_new_random_walk):
                idx_random_walk = (
                    np.flatnonzero(countdown_zero)[is_new_random_walk]
                )

                # Find currently active random_walk baboons
                existing_random_walkers = np.flatnonzero(
                    next_state.state == State.random_walk.value
                )

                for idx in idx_random_walk:
                    if (
                        choose_drift_from_other_random_walkers
                        and existing_random_walkers.size > 0
                    ):
                        # Pick a random existing random walker
                        chosen_idx = rng.choice(existing_random_walkers)
                        base_drift = next_state.random_walk_drift[chosen_idx]

                        # Perturb angle
                        base_angle = np.arctan2(base_drift[1], base_drift[0])
                        perturbed_angle = (
                            base_angle
                            + rng.normal(0, new_random_walk_drift_angle_std)
                        )
                        # Keep the same magnitude (step size with slight noise)
                        step_length = (
                            np.linalg.norm(base_drift)
                            + rng.normal(0, following_step_length_std)
                        )

                        drift = np.array([
                            np.cos(perturbed_angle),
                            np.sin(perturbed_angle),
                        ]) * step_length
                    else:
                        # If no existing random_walk baboons, random drift
                        random_angle = rng.uniform(0, 2 * np.pi)
                        step_length = (
                            random_walk_step_length
                            + rng.normal(0, random_walk_step_length_std)
                        )
                        drift = np.array([
                            np.cos(random_angle),
                            np.sin(random_angle),
                        ]) * step_length

                    next_state.random_walk_drift[idx] = drift

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

            # Get all pairwise angles and distances
            all_angles = get_angles(current_positions)
            all_distances = get_distances(current_positions)

            # Extract the angle and distance from follower to its target
            base_angles = all_angles[idx_following, targets]
            distances = all_distances[idx_following, targets]

            # Perturb angles
            perturbed_angles = base_angles + rng.normal(
                0, angle_std, size=base_angles.shape[0]
            )

            # Compute directions from perturbed angles
            directions = np.column_stack((
                np.cos(perturbed_angles),
                np.sin(perturbed_angles)
            ))

            # Step size depends on distance + noise, and is clipped
            step_lengths = np.clip(
                (
                    (
                        following_step_length_proportion
                        * (distances - following_radius)
                    ) + rng.normal(
                        0, following_step_length_std, size=distances.shape[0],
                    )
                ),
                -max_follow_step,
                max_follow_step,
            )

            drift_vectors[idx_following] = (
                directions * step_lengths[:, np.newaxis]
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
