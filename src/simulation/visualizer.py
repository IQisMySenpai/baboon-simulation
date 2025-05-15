import matplotlib.pyplot as plt
from typing import Optional, Sequence
import numpy as np
from matplotlib.animation import FuncAnimation
from simulation.sim_output import SimOutput
import numpy.typing as npt
from matplotlib import collections


class PointVisualizer(SimOutput):
    def __init__(
        self,
        xlim=(-400, 400),
        ylim=(-400, 400),
        figsize=(8, 8),
    ):
        """
        Initialize the 2D point visualizer.

        Args:
            xlim: x-axis limits
            ylim: y-axis limits
            figsize: Figure size
        """
        self.xlim = xlim
        self.ylim = ylim

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        # Initialize scatter plot
        self.scat = self.ax.scatter(
            [],
            [],
            s=100,
            facecolors=[],  # inside color
            edgecolors=[] ,  # border color
            marker='.',  # either 'o' or '.', depending what size you want
        )

    def animate(
        self,
        baboons_trajectory: npt.NDArray[float],
        colors: Optional[Sequence[str]],
        interval: int = 1000,
    ):
        """
        Set up the animation for the scatter plot using the stored positions
        and colors.

        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            interval: Delay between frames in milliseconds
        """

        def update_frame(num):
            self.scat.set_offsets(baboons_trajectory[num, :, :])
            if colors:
                self.scat.set_facecolor(colors)
                self.scat.set_edgecolor(colors)

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
        colors: Optional[Sequence[str]],
        filename: str,
        fps: int = 30,
        file_format: str = "mp4",
        dpi: int = 300,
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
            dpi: Dots per inch (default is 300)
        """
        animation = self.animate(baboons_trajectory, colors)
        animation.save(
            f"{filename}.{file_format}", writer="ffmpeg", fps=fps, dpi=dpi,
        )
        print(f"Animation saved to {filename}")


class LineVisualizer (SimOutput):
    def __init__(
        self,
        xlim=(-400, 400),
        ylim=(-400, 400),
        figsize=(8, 8),
    ):
        """
        Initialize the 2D line visualizer.
        Args:
            xlim: x-axis limits
            ylim: y-axis limits
            figsize: Figure size
        """
        self.xlim = xlim
        self.ylim = ylim

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.grid()

        self.lines = []  # One line per baboon
        self.xdata = []
        self.ydata = []

    def animate(
        self,
        baboons_trajectory: npt.NDArray[np.float64],
        colors: Optional[Sequence[str]] = None,
        interval: int = 100,
    ):
        """
        Set up the animation for the line plot using the stored positions
        and colors.
        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            interval: Delay between frames in milliseconds
        """
        n_baboons = baboons_trajectory.shape[1]

        # Initialize lines and data containers
        self.xdata = [[] for _ in range(n_baboons)]
        self.ydata = [[] for _ in range(n_baboons)]

        # Initialize lines for each baboon
        for i in range(n_baboons):
            (line,) = self.ax.plot([], [], lw=2, color=colors[i] if colors else None)
            self.lines.append(line)

        def init():
            for line in self.lines:
                line.set_data([], [])
            return self.lines

        def update_frame(frame_idx):
            for i, line in enumerate(self.lines):
                self.xdata[i].append(baboons_trajectory[frame_idx, i, 0])
                self.ydata[i].append(baboons_trajectory[frame_idx, i, 1])
                line.set_data(self.xdata[i], self.ydata[i])
            return self.lines

        self.anim = FuncAnimation(
            self.fig,
            update_frame,
            frames=baboons_trajectory.shape[0],
            init_func=init,
            interval=interval,
            blit=True,
            repeat=False,
        )

        return self.anim

    def save(
        self,
        baboons_trajectory: npt.NDArray[np.float64],
        colors: Optional[Sequence[str]],
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
        anim = self.animate(baboons_trajectory, colors)
        anim.save(f"{filename}.{file_format}", writer="ffmpeg", fps=fps, dpi=300)
        print(f"Animation saved to {filename}.{file_format}")


class TrajectoryVisualizer (SimOutput):
    def __init__(
        self,
        xlim=(-400, 400),
        ylim=(-400, 400),
        figsize=(8, 8),
    ):
        """
        Initialize the 2D trajectory visualizer.
        """
        self.xlim = xlim
        self.ylim = ylim

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def plot(
        self,
        baboons_trajectory: np.ndarray,
        colors: Optional[Sequence[str]],
        linewidth: float = 2.0,
    ):
        """
        Plot the full trajectory of each baboon as a line.

        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            linewidth: Width of the lines
        """
        n_steps, n_baboons, _ = baboons_trajectory.shape

        segments = []
        for baboon_idx in range(n_baboons):
            path = baboons_trajectory[:, baboon_idx, :]
            segments.append(path)

        line_collection = collections.LineCollection(
            segments,
            colors=colors,
            linewidths=linewidth,
        )
        self.ax.add_collection(line_collection)
        self.ax.autoscale_view()

    def save(
        self,
        baboons_trajectory: np.ndarray,
        colors: Optional[Sequence[str]],
        filename: str,
        file_format: str = "png",
    ):
        """
        Save the static trajectory plot as an image.

        Args:
            baboons_trajectory: Full trajectory of baboons.
                Shape (#steps + 1, n_baboons, 2)
            colors: List of the colors of each baboon. Length n_baboons.
            filename: The name of the output file
            fps: Not used in static version, kept for compatibility
            file_format: File format (e.g., 'png', 'jpg', 'svg')
        """
        # Plot the trajectory
        self.plot(baboons_trajectory, colors)

        # Save the figure as an image
        full_filename = f"{filename}.{file_format}"
        self.fig.savefig(full_filename, dpi=300, format=file_format)
        print(f"Image saved to {full_filename}")