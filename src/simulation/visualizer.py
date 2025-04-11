import matplotlib.pyplot as plt
from typing import Sequence
import numpy as np
from matplotlib.animation import FuncAnimation
from simulation.sim_output import SimOutput
from matplotlib.colors import to_hex


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

        # Store the positions and colors for animation
        self.positions_list = []
        self.colors_list = []

    def update(self, positions: np.ndarray, colors: Sequence[str]):
        """
        Update the scatter plot with new positions (and optional colors).
        The updated positions and colors are saved for animation.

        :param positions: (N, 2) array-like of x and y coordinates
        :param colors: Optional list of colors (length N)
        """
        self.scat.set_offsets(positions)
        self.scat.set_color(colors)

        # Save the current positions and colors for later animation
        self.positions_list.append(positions)
        self.colors_list.append(colors)

    def animate(self, interval: int = 100):
        """
        Set up the animation for the scatter plot using the stored positions and colors.

        :param interval: Delay between frames in milliseconds
        """

        def update_frame(num):
            positions = self.positions_list[num]
            colors = self.colors_list[num]

            self.scat.set_offsets(positions)
            self.scat.set_color(colors)

        # Set up the animation
        ani = FuncAnimation(
            self.fig,
            update_frame,
            frames=len(self.positions_list),
            interval=interval,
            repeat=False,
        )

        return ani

    def show(self):
        """
        Display the animation by calling animate.
        """
        ani = self.animate()  # Get the animation
        plt.show()

    def save(self, filename: str, fps: int = 30, file_format: str = 'mp4'):
        """
        Save the animation to a file (e.g., MP4 or GIF).

        :param filename: The name of the output file
        :param fps: Frames per second (default is 30)
        :param file_format: File format (default is 'mp4', other options include 'gif', 'avi', etc.)
        """
        ani = self.animate()  # Get the animation
        ani.save(filename, writer='ffmpeg', fps=fps, dpi=300)  # Saving as MP4
        print(f"Animation saved to {filename}")


