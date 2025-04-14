import numpy as np
from simulation.simulator import Simulator
from simulation.visualizer import PointVisualizer

# Define your baboons
baboons = np.array([
    [10, 20],
    [50, 70],
    [30, 40],
    [60, 80],
])

colors = ["red", "blue", "green", "yellow"]

# Create a Simulator instance
simulator = Simulator(total_steps=500, baboons=baboons)

# Create an output visualizer (assuming SimOutput implements the necessary interface)
visualizer = PointVisualizer()

# Run the simulation with visual output
simulator.run(output=visualizer)