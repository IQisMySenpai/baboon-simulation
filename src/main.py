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
    seed = 93538478
    total_time_steps = 2000
    n_baboons = 15  # 33
    scale = 30  # std of the normal distribution to generate initial positions
    np.random.seed(seed)
    initial_baboons = np.random.normal(0, scale, (n_baboons, 2))
    diffusion_constant = 4

    # Drift parameters and function
    drift_diffusion_with_state = state_driven_drift_diffusion_function(
        angle_std=30 * np.pi / 180,
        group_influence_step_length=0.2,
        random_walk_step_length=0.3,
        random_walk_step_length_std=0.2,
        min_follow_distance=5.0,
        max_follow_step=0.3,
        following_step_length_std=0.2,
        following_step_length_proportion=0.25,
        following_radius=20.0,
        choose_drift_from_other_random_walkers=True,
        new_random_walk_drift_angle_std=10 * np.pi / 180,
        state_diffusion_constants={
            State.still: 0.15,
            State.following: 0.2,
            State.group_influence: 0.2,
            State.random_walk: 0.3,
        },
        state_probabilities={  # probability to choose each state
            State.following: 0.65,
            State.group_influence: 0,
            State.still: 0.05,
            State.random_walk: 0.3,
        },
        probability_repeat_random_walk=0.90,
        state_countdown_means={  # average time-steps in each state
            State.following: 100,
            State.group_influence: 0,
            State.still: 0,
            State.random_walk: 300,
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
