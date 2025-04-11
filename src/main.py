from map_entities.baboon import Baboon
from simulation.simulator import Simulator
from simulation.visualizer import PointVisualizer

# Define your baboons
baboons = [
    Baboon(x=10, y=20, color="red"),
    Baboon(x=50, y=70, color="blue"),
    Baboon(x=30, y=40, color="green"),
    Baboon(x=60, y=80, color="yellow"),
]

# Create a Simulator instance
simulator = Simulator(total_steps=500, baboons=baboons)

# Create an output visualizer (assuming SimOutput implements the necessary interface)
visualizer = PointVisualizer()

# Run the simulation with visual output
simulator.run(output=visualizer)