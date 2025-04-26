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
from strategies.drift_with_states import (
    state_driven_drift_diffusion_function,
    State,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "../outputs")

if __name__ == "__main__":

    # ######################## SIMULATION PARAMETERS ##########################
    seed = 0
    total_time_steps = 1000
    n_baboons = 10
    scale = 5  # std of the normal distribution to generate initial positions
    np.random.seed(136873234)
    initial_baboons = np.random.normal(0, scale, (n_baboons, 2))
    diffusion_constant = 4

    # Drift parameters and function
    drift_diffusion_with_state = state_driven_drift_diffusion_function(
        angle_std=0,
        group_influence_step_length=0.3,
        random_walk_step_length=0.35,
        min_follow_distance=10.0,
        min_follow_step=0.1,
        max_follow_step=0.6,
        following_step_size_std=0.4,
        following_step_size_proportion=0.25,
        state_diffusion_constants={
            State.still: 0.05,
            State.following: 0.2,
            State.group_influence: 0.2,
            State.random_walk: 2,
        },
        state_probabilities={  # probability to choose each state
            State.following: 0.4,
            State.group_influence: 0.4,
            State.still: 0.1,
            State.random_walk: 0.1,
        },
        state_countdown_means={  # mean time-steps in each state
            State.following: 30,
            State.group_influence: 12,
            State.still: 8,
            State.random_walk: 50,
        },
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

    visualizer = PointVisualizer()
    visualizer.save(
        baboons_trajectory=baboons_trajectory,
        colors=colors,
        filename=os.path.join(OUT_DIR, "baboons_visualization"),
    )
