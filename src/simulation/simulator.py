from typing import List, Optional
from map_entities.baboon import Baboon
from map_entities.point2d import Point2D
import numpy as np
import random
from tqdm import tqdm
from simulation.sim_output import SimOutput


class Simulator:
    def __init__(self, total_steps: int, seed: int = 42, baboons: Optional[List[Baboon]] = None):
        """
        Initialize the simulator with a given number of simulation steps.
        :param total_steps: Total number of simulation steps
        """

        self.baboons: List[Baboon] = []
        self.current_step = 0
        self.total_steps = total_steps

        # Set the random seed for reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize baboons list (if not provided)
        self.baboons = baboons if baboons else []

    def calculate_baboon_move(self, baboon: Baboon) -> np.ndarray:
        return np.random.uniform(-1, 1, size=2)

    def get_baboon_positions(self) -> np.ndarray:
        return np.array([b.coordinates for b in self.baboons])

    def get_baboon_colors(self) -> list:
        return [b.color for b in self.baboons]

    def step(self):
        """
        Perform a single simulation step.
        """
        self.current_step += 1

        moves : List[np.ndarray] = []

        # Calculate the move for each baboon so that we can update them all at once
        for baboon in self.baboons:
            moves.append(self.calculate_baboon_move(baboon))

        assert len(moves) == len(self.baboons), "The number of moves should match the number of baboons"

        # Update the position of each baboon
        for i in range(len(self.baboons)):
            baboon = self.baboons[i]
            # Update the baboon's position
            baboon.move(moves[i])

    def run(self, output: Optional[SimOutput] = None):
        """
        Run the simulation for the specified number of steps.
        """
        for i in tqdm(range(self.total_steps), desc="Simulation Progress"):
            assert self.current_step < self.total_steps, "Simulation has already completed"
            assert self.current_step == i, "Current step should match the loop index"

            # Perform the simulation step
            self.step()

            if output:
                output.update(positions= self.get_baboon_positions(), colors=self.get_baboon_colors())

        if output:
            output.save('output.mp4')
