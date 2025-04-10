import numpy as np

class Point2D (np.ndarray):
    """
    A class representing a 2D point in space.
    Inherits from numpy's ndarray to leverage its array functionalities.
    """

    def __init__(self, x: float, y: float, color: str = "black"):
        """
        Initialize a 2D point with x, y coordinates and a color.
        :param x: X coordinate
        :param y: Y coordinate
        :param color: Color of the point, default is "black"
        """
        super().__init__((x, y))
        self.color = color

    def __str__(self):
        return f"Point2D({self[0]}, {self[1]}, color={self.color})"

    def __repr__(self):
        return f"Point2D({self[0]}, {self[1]}, color={self.color})"