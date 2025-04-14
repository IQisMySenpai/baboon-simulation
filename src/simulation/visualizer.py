import matplotlib.pyplot as plt
from typing import Sequence
import numpy as np
from matplotlib.animation import FuncAnimation
from simulation.sim_output import SimOutput
import numpy.typing as npt


class PointVisualizer(SimOutput):
    def __init__(
        self,
        xlim=(0, 100),
        ylim=(0, 100),
        figsize=(6, 6),
    ):
        """
        Initialize the 2D point visualizer.

        :param xlim: Tuple of x-axis limits
        :param ylim: Tuple of y-axis limits
        :param figsize: Size of the matplotlib figure
        """
        self.xlim = xlim
        self.ylim = ylim

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        # Initialize scatter plot
        self.scat = self.ax.scatter([], [], c=[], s=30)

    def animate(
        self,
        baboons_trajectory: npt.NDArray[float],
        colors: Sequence[str],
        interval: int = 100,
    ):
        """
        Set up the animation for the scatter plot using the stored positions and colors.

        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            interval: Delay between frames in milliseconds
        """

        def update_frame(num):
            self.scat.set_offsets(baboons_trajectory[num, :, :])
            self.scat.set_color(colors)

        # Set up the animation
        animation = FuncAnimation(
            self.fig,
            update_frame,
            frames=baboons_trajectory.shape[0],
            interval=interval,
            repeat=False,
        )

        return animation

    def save(
        self,
        baboons_trajectory: npt.NDArray[np.float64],
        colors: Sequence[str],
        filename: str,
        fps: int = 30,
        file_format: str = "mp4",
    ):
        """
        Save the animation to a file (e.g., MP4 or GIF).

        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            filename: The name of the output file
            fps: Frames per second (default is 30)
            file_format: File format (default is "mp4", other options include
                "gif", "avi", etc.)
        """
        animation = self.animate(baboons_trajectory, colors)
        animation.save(
            f"{filename}.{file_format}", writer="ffmpeg", fps=fps, dpi=300,
        )
        print(f"Animation saved to {filename}")


