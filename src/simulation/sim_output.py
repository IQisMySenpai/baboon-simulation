from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence


class SimOutput(ABC):
    """
    Abstract base class that defines the required interface for loggers.
    Any class that inherits from Logger must implement `save`.
    """

    @abstractmethod
    def save(self, filename: str):
        """
        Save the current plot to a file.
        """
        pass