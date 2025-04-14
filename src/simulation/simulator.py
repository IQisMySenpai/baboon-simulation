from typing import Iterable, List, Optional
import numpy.typing as npt
from map_entities.baboon import Baboon
import numpy as np
from tqdm import tqdm
from simulation.sim_output import SimOutput
from supporting.angular_distribution import (
    angular_distribution, sample_from_distribution
)


class Simulator:
    def __init__(
        self,
        total_steps: int,
        initial_baboons: npt.NDArray[np.float64],
    ):
        """
        Initialize the simulator with a given number of simulation steps.

        Args:
            total_steps: Total number of simulation steps
            seed: Random seed for reproducibility
            initial_baboons: (n_baboons, 2)-np.array with initial positions
                of baboons. 
        """

        self.baboons: List[Baboon] = []
        self.current_step = 0
        self.total_steps = total_steps

        # Initialize baboons list (if not provided)
        self.baboons = baboons if baboons else list()

    def calculate_baboon_move(self, baboon: Baboon) -> np.ndarray:
        angles = []
        sigmas = []

        for other_baboon in self.baboons:
            if other_baboon != baboon:
                angles.append(baboon.angle(other_baboon))
                sigmas.append(min(0.5, max(0.1, baboon.distance(other_baboon) / 20)))

        theta_grid, probs = angular_distribution(angles, sigmas)
        direction = sample_from_distribution(self.rng, theta_grid, probs)

        # Calculate the new position based on the direction
        move = np.ndarray((2,), dtype=float)
        move[0], move[1] = np.cos(direction) * 0.1, np.sin(direction) * 0.1 # TODO: 0.1 is the step size

        return move

    def get_baboon_positions(self) -> np.ndarray:
        return np.array([b.coordinates for b in self.baboons])

    def get_baboon_colors(self) -> list:
        return [b.color for b in self.baboons]

    def step(self):
        """
        Perform a single simulation step.
        """
        self.current_step += 1

        moves: List[np.ndarray] = []

        # Calculate the move for each baboon so that we can update them all at once
        for baboon in self.baboons:
            moves.append(self.calculate_baboon_move(baboon))

        assert len(moves) == len(self.baboons), "The number of moves should match the number of baboons"

        # Update the position of each baboon
        for i in range(len(self.baboons)):
            baboon = self.baboons[i]
            # Update the baboon's position
            baboon.move(moves[i])

    def run(
        self,
        output: Optional[SimOutput] = None,
        seed: Optional[int] = None,
    ):
        """
        Run the simulation for the specified number of steps.
        """
        # Use random generator for reproducibility
        # recommended for numpy rather thatn using random.seed
        self.rng = np.random.default_rng(seed)  # Create a random generator

        for i in tqdm(range(self.total_steps), desc="Simulation Progress"):
            assert self.current_step < self.total_steps, "Simulation has already completed"
            assert self.current_step == i, "Current step should match the loop index"

            # Perform the simulation step
            self.step()

            if output:
                output.update(positions=self.get_baboon_positions(), colors=self.get_baboon_colors())

        if output:
            output.save('../outputs/output.mp4')
