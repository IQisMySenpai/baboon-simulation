import numpy as np
from enum import Enum
from sklearn.utils import Bunch
import numpy.typing as npt
from typing import Callable, Optional, Tuple
from simulation_types.documentation import DriftDiffusionWithStateType
from utils.baboons import get_angles, get_differences, get_distances


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
        - target: The baboon is moving towards a target in the targets array.
    """
    following = 1
    group_influence = 2
    still = 3
    random_walk = 4
    target = 5


def state_driven_drift_diffusion_with_targets_function(
    angle_std: float,
    group_influence_step_length: float,
    random_walk_step_length: float,
    random_walk_step_length_std: float,
    min_follow_distance: float,
    max_follow_distance: float,
    max_follow_step: float,
    state_diffusion_constants: dict[State, float],
    targets: npt.NDArray[np.float64],
    following_step_length_std: float,
    following_step_length_proportion: float,
    following_radius: float,
    target_radius: float,
    n_max_targets: int = 7,
    state_probabilities: Optional[dict[State, float]] = None,
    state_countdown_means: Optional[dict[State, float]] = None,
    probability_repeat_random_walk: float = 0.0,
    choose_drift_from_other_random_walkers: bool = True,
    new_random_walk_drift_angle_std: float = 10 * np.pi / 180,
    new_target_noise_std: float = 5.0,
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
        targets (npt.NDArray[np.float64]): Array of target
            coordinates (n_targets, 2). If provided, target-state baboons will
            move towards these targets as a drift vector. The target is chosen
            if it is the closest to the baboon's current position. After a
            target is chosen by a target-state baboon, only following baboons
            which have not come close enough to the target will be able to
            choose this target-state baboon it as a following target.

            Logic is as follows:
            - If a baboon is in target-state, it will choose the first
                target and move towards it until it is close enough
                (determined by the baboon's target_radius).
                After it is close enough to the target, it will select next
                target from the list of targets and move towards it (and so
                on until the target-state runs out). Next time that the
                baboon is in target-state, it will choose the first target
                that it has not yet visited (i.e. gotten close to according
                to target_radius).
            - If a baboon is in following-state, it will choose to follow
                baboons that are in target-state or random_walk-state but
                discarding baboons in target-state which are going towards
                target that the follower has already yet visited (i.e. the
                follower did not yet get close enough to yet according to
                target_radius). If no baboon is available to follow, the
                baboon will become target-state.
            At the end fo the loop, the state will keep an array of
            visited_targets = np.array((n_baboons, n_targets), dtype=bool)
            where each baboon will have a boolean array of size n_targets
            indicating whether it has already visited the target or not (i.e.
            gotten closer to it than target_radius).

    Returns:
        DriftDiffusionWithStateType: A callable that computes the drift vector,
        diffusion matrix, and updated internal state for all baboons given
        their trajectory history and a random generator.
    """

    if state_probabilities is None:
        state_probabilities = {
            State.following: 0.2,
            State.group_influence: 0.2,
            State.still: 0.2,
            State.random_walk: 0.2,
            State.target: 0.2,
        }
    if state_countdown_means is None:
        state_countdown_means = {
            State.following: 20,
            State.group_influence: 20,
            State.still: 20,
            State.random_walk: 20,
            State.target: 20,
        }
    state_list = list(state_probabilities.keys())
    prob_list = list(state_probabilities.values())
    original_targets = targets.copy()

    def drift_diffusion(
        baboons_trajectory: npt.NDArray[np.float64],
        rng: np.random.Generator,
        state_bunch: Optional[Bunch],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Bunch]:
        n_baboons = baboons_trajectory.shape[1]
        current_positions = baboons_trajectory[-1, :, :]  # (n_baboons, 2)

        if state_bunch is None:  # initial state
            state_bunch = Bunch(
                state=np.full(n_baboons, State.still.value),
                following_idx=np.arange(n_baboons),
                state_countdown=np.zeros(n_baboons, dtype=int),
                random_walk_drift=np.zeros((n_baboons, 2)),
                targets=original_targets,
                visited_targets=np.zeros(
                    (n_baboons, original_targets.shape[0]),
                    dtype=bool,
                ),
            )

        next_state = Bunch(
            state=state_bunch.state.copy(),
            following_idx=state_bunch.following_idx.copy(),
            state_countdown=state_bunch.state_countdown.copy(),
            random_walk_drift=state_bunch.random_walk_drift.copy(),
            targets=state_bunch.targets.copy(),
            visited_targets=state_bunch.visited_targets.copy(),
        )

        # Update visited_targets
        target_deltas = (  # (n_baboons, n_targets, 2)
            current_positions[:, np.newaxis, :]
            - next_state.targets[np.newaxis, :, :]
        )
        # (n_baboons, n_targets):
        distances_to_targets = np.linalg.norm(target_deltas, axis=2)
        next_state.visited_targets = np.logical_or(
            next_state.visited_targets,
            (distances_to_targets < target_radius),
        )
        # Add new target if some baboon has visited all
        # if n_max_targets has not been reached
        all_visited_mask = np.all(next_state.visited_targets, axis=1)
        if (
            np.any(all_visited_mask)
            and n_max_targets > next_state.targets.shape[0]
        ):
            # Compute the new target position
            direction = next_state.targets[-1] - next_state.targets[-2]
            noise = rng.normal(0, new_target_noise_std, size=direction.shape)
            new_target = next_state.targets[-1] + direction + noise

            # Append new target to targets array
            next_state.targets = np.vstack([next_state.targets, new_target])

            # Expand visited_targets array to accommodate new target
            n_baboons = next_state.visited_targets.shape[0]
            new_column = np.zeros((n_baboons, 1), dtype=bool)
            next_state.visited_targets = np.hstack([
                next_state.visited_targets, new_column,
            ])

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
                    # Exclude too close or too far baboons
                    in_range = (
                        (distances > min_follow_distance)
                        & (distances < max_follow_distance)
                    )
                    # Candidate baboons: either random_walk OR target_state
                    # where target not yet visited
                    valid_non_target = (
                        next_state.state == State.random_walk.value
                    ) | (
                        next_state.state == State.still.value
                    )
                    valid_target = next_state.state == State.target.value

                    # Get which target each baboon in target mode is pursuing
                    # (first unvisited)
                    first_unvisited_indices = np.argmax(
                        ~next_state.visited_targets, axis=1  # per baboon
                    )
                    target_baboons = np.flatnonzero(valid_target)
                    pursuing_indices = first_unvisited_indices[target_baboons]
                    # For baboons in target-state: are they pursuing a target
                    # the follower hasn't visited?
                    follower_visited = next_state.visited_targets[idx]
                    still_relevant_targets = (
                        ~follower_visited[pursuing_indices]
                    )
                    valid_target_indices = (
                        target_baboons[still_relevant_targets]
                    )
                    possible_targets = np.flatnonzero(
                        in_range & (valid_non_target | np.isin(
                            np.arange(n_baboons), valid_target_indices,
                        ))
                    )

                    if possible_targets.size > 0:
                        target_distances = distances[possible_targets]
                        weights = np.exp(-target_distances)
                        next_state.following_idx[idx] = rng.choice(
                            possible_targets, p=weights / weights.sum(),
                        )
                    else:
                        # No one to follow -> become a target-state baboon
                        next_state.state[idx] = State.target.value
                        new_states[i] = State.target.value
                        next_state.following_idx[idx] = 0
                elif new_states[i] == State.target.value:
                    # Nothing to sample here, target logic happens below
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
            print("# of GROUP INFLUENCE: ", np.sum(is_group_influence))
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

        # ========== TARGET ==========
        is_target_state = next_state.state == State.target.value
        if np.any(is_target_state):
            idx_target = np.flatnonzero(is_target_state)

            # Each target-state baboon chooses the first unvisited target
            target_indices = np.argmax(
                ~next_state.visited_targets[idx_target], axis=1,
            )  # (n_target_baboons,)
            target_positions = next_state.targets[target_indices]

            # Compute angles and distances from baboons to their targets
            deltas = target_positions - current_positions[idx_target]
            base_angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            distances = np.linalg.norm(deltas, axis=1)

            # Perturb angles
            perturbed_angles = base_angles + rng.normal(
                0, angle_std, size=base_angles.shape[0]
            )

            # Convert angles to directions
            directions = np.column_stack((
                np.cos(perturbed_angles),
                np.sin(perturbed_angles)
            ))

            # Step size proportional to (distance - target_radius) + noise
            step_lengths = np.clip(
                (
                    following_step_length_proportion
                    * (distances - (target_radius) * 0.5)
                    + rng.normal(
                        0, following_step_length_std, size=distances.shape[0],
                    )
                ),
                -max_follow_step,
                max_follow_step,
            )

            drift_vectors[idx_target] = (
                directions * step_lengths[:, np.newaxis]
            )

            # Update diffusion matrices
            diffusion_matrices[is_target_state, :, :] = (
                np.eye(2) * state_diffusion_constants[State.target]
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
