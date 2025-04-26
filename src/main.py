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
    total_time_steps = 400
    n_baboons = 10  # 33
    scale = 40  # std of the normal distribution to generate initial positions
    np.random.seed(136873234)
    initial_baboons = np.random.normal(0, scale, (n_baboons, 2))
    diffusion_constant = 4

    # Drift parameters and function
    drift_diffusion_with_state = state_driven_drift_diffusion_function(
        angle_std=0,
        step_length=0.4,
        max_follow_step=0.5,
        countdown_poisson_mean=20,
        sigma_still=0.01,
        sigma_following=0.1,
        sigma_group_influence=0.1,
        sigma_random_walk=0.5,
        following_step_size_std=0.05,
        following_step_size_proportion=0.1,
        state_probabilities={
            State.following: 0.1,
            State.group_influence: 0.2,
            State.still: 0.1,
            State.random_walk: 0.6,
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
