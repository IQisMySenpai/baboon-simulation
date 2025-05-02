from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class SimOutput(ABC):
    """
    Abstract base class that defines the required interface for loggers.
    Any class that inherits from Logger must implement `save`.
    """

    @abstractmethod
    def save(
        self,
        baboons_trajectory: npt.NDArray[np.float64],
        filename: str,
        **kwargs,
    ):
        """
        Save the current plot to a file.
        """
        pass