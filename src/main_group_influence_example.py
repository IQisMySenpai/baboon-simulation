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
    seed = 92765478
    total_time_steps = 3000
    n_baboons = 15  # 33
    scale = 30  # std of the normal distribution to generate initial positions
    np.random.seed(seed)
    initial_baboons = np.random.normal(0, scale, (n_baboons, 2))
    diffusion_constant = 4

    def bm_drift_function(t):
        """
        2-dim drift for the Brownian motion that drives the baboon equation.
        """
        drift_direction = np.array([
            np.cos(t / 300),
            np.sin(t / 300),
        ])
        drift_strength = 10 * np.sin(t / 20)
        return drift_direction * drift_strength

    bm_drift_function = None
    # Drift parameters and function
    drift_diffusion_with_state = state_driven_drift_diffusion_function(
        angle_std=30 * np.pi / 180,
        group_influence_step_length=0.2,
        random_walk_step_length=0.3,
        random_walk_step_length_std=0.2,
        min_follow_distance=0.0,
        max_follow_distance=150.0,
        max_follow_step=0.3,
        following_step_length_std=0.2,
        following_step_length_proportion=0.25,
        following_radius=20.0,
        choose_drift_from_other_random_walkers=True,
        new_random_walk_drift_angle_std=30 * np.pi / 180,
        state_diffusion_constants={
            State.still: 0.05,
            State.following: 0.05,
            State.group_influence: 0.05,
            State.random_walk: 0.05,
        },
        state_probabilities={  # probability to choose each state
            State.following: 0,  # 0.95,
            State.group_influence: 1,
            State.still: 0,
            State.random_walk: 0,  # 0.05,
        },
        probability_repeat_random_walk=0.90,
        state_countdown_means={  # average time-steps in each state
            State.following: 80,
            State.group_influence: 20000,
            State.still: 0,
            State.random_walk: 500,
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
        bm_drift_function=bm_drift_function,
    )
    baboons_trajectory = simulator.run()
    lim = 100
    visualizer = PointVisualizer(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
    )
    visualizer.save(
        baboons_trajectory=baboons_trajectory[:400],  # Save every x-th frame
        colors=colors,
        filename=os.path.join(OUT_DIR, "group_influence"),
        fps=25,
        file_format="gif",
        dpi=100,
    )
