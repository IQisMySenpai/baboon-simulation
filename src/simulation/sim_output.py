from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence

class SimOutput(ABC):
    """
    Abstract base class that defines the required interface for loggers.
    Any class that inherits from Logger must implement `update`, `show`, and `visualize`.
    """

    @abstractmethod
    def update(self, positions: np.ndarray, colors: Sequence[str]):
        """
        Update the visualization with new positions (and optional colors).
        :param positions: (N, 2) array-like of x and y coordinates
        :param colors: Optional list of colors (length N)
        """
        pass

    @abstractmethod
    def show(self):
        """
        Display the visualization (blocking).
        """
        pass

    @abstractmethod
    def save(self, filename: str):
        """
        Save the current plot to a file.
        """
        pass