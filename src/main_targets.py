# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from simulation.simulator import Simulator
from simulation.visualizer import PointVisualizer
from strategies.drift import (
    only_angles_drift_function,
)
from strategies.diffusion import (
    constant_diffusion_function,
)
from strategies.drift_with_states_and_targets import (
    state_driven_drift_diffusion_with_targets_function,
    State,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "../outputs")

if __name__ == "__main__":

    # ######################## SIMULATION PARAMETERS ##########################
    seed = 92213214
    total_time_steps = 16000
    n_baboons = 16  # 33
    scale = 200  # std of the normal distribution to generate initial positions
    np.random.seed(seed)
    initial_baboons = (
        np.array([-100, -100])
        + np.random.normal(0, scale, (n_baboons, 2))
    )
    diffusion_constant = 4
    targets = np.array([
        [100, 100],
        [150, 300],
        [350, 400],
        [360, 200],
        [200, 000],
    ]) * 2

    # Drift parameters and function
    drift_diffusion_with_state = (
        state_driven_drift_diffusion_with_targets_function(
            angle_std=50 * np.pi / 180,
            group_influence_step_length=0.4,
            random_walk_step_length=0.4,
            random_walk_step_length_std=0.05,
            min_follow_distance=0.0,
            max_follow_distance=150.0,
            max_follow_step=0.4,
            following_step_length_std=0.05,
            following_step_length_proportion=0.25,
            following_radius=20.0,
            target_radius=30.0,
            choose_drift_from_other_random_walkers=False,
            new_random_walk_drift_angle_std=30 * np.pi / 180,
            targets=targets,
            new_target_noise_std=5.0,
            n_max_targets=5,
            state_diffusion_constants={
                State.still: 0.2,
                State.following: 0.2,
                State.group_influence: 0.2,
                State.random_walk: 0.2,
                State.target: 0.2,
            },
            probability_repeat_random_walk=0,
            state_probabilities={  # probability to choose each state
                State.following: 0.4,
                State.group_influence: 0.05,
                State.still: 0.2,
                State.random_walk: 0.05,
                State.target: 0.3,
            },
            state_countdown_means={  # average time-steps in each state
                State.following: 200,
                State.group_influence: 200,
                State.still: 500,
                State.random_walk: 200,
                State.target: 250,
            },
        )
    )
    # #################### END OF SIMULATION PARAMETERS #######################

    # Generate a list of colors based on the number of baboons
    colors = plt.get_cmap("tab20", n_baboons)
    # Convert the colormap to a list of RGB values
    colors = [colors(i) for i in range(n_baboons)]

    # Simulation and visualization
    simulator = Simulator(
        total_time_steps=total_time_steps,
        initial_baboons=initial_baboons,
        dt=1,
        drift=None,
        diffusion=None,
        seed=seed,
        drift_diffusion_with_state=drift_diffusion_with_state,
    )
    baboons_trajectory = simulator.run()

    visualizer = PointVisualizer(
        xlim=(-400, 1000),
        ylim=(-400, 1000),
    )
    visualizer.save(
        baboons_trajectory=baboons_trajectory[1000:11000:40],  # Save every x-th frame
        colors=colors,
        filename=os.path.join(OUT_DIR, "target_based"),
        fps=10,
        file_format="gif",
        dpi=100,
    )
