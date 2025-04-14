import os
import numpy as np
from simulation.simulator import Simulator
from simulation.visualizer import PointVisualizer
from strategies.drift import (
    only_angles_drift_function,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "../outputs")

if __name__ == "__main__":

    # ######################## SIMULATION PARAMETERS ##########################
    seed = 0
    total_time_steps = 200
    initial_baboons = np.array([
        [10, 20],
        [50, 70],
        [30, 40],
        [60, 80],
    ])
    colors = ["red", "blue", "green", "yellow"]

    # Drift parameters and function
    angle_std = 1  # standard deviation of the normal perturbation of the angle
    step_length = 0.4  # the size of the steps of the baboons
    drift = only_angles_drift_function(
        angle_std=angle_std,
        step_length=step_length,
    )
    # #################### END OF SIMULATION PARAMETERS #######################



    # Simulation and visualization
    simulator = Simulator(
        total_time_steps=total_time_steps,
        initial_baboons=initial_baboons,
        dt=1,
        drift=drift,
        diffusion=None,
        seed=seed,
    )
    baboons_trajectory = simulator.run()

    visualizer = PointVisualizer()
    visualizer.save(
        baboons_trajectory=baboons_trajectory,
        colors=colors,
        filename=os.path.join(OUT_DIR, "baboons_visualization"),
    )
