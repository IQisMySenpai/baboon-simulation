import numpy as np

class Point2D:
    """
    A class representing a 2D point in space.
    """

    def __init__(self, x: float, y: float, color: str = "black"):
        """
        Initialize a Point2D object.
        :param x: X coordinate
        :param y: Y coordinate
        :param color: Color of the point (default is "black")
        """
        self.color = color
        self._coordinates = np.ndarray((2,), dtype=float)
        self._coordinates[0] = x
        self._coordinates[1] = y

    def __str__(self):
        return f"Point2D({self._coordinates[0]}, {self._coordinates[1]}, color={self.color})"

    def __repr__(self):
        return f"Point2D({self._coordinates[0]}, {self._coordinates[1]}, color={self.color})"

    def __eq__(self, other):
        """
        Check if two Point2D objects are equal based on their coordinates.
        :param other: Another Point2D object
        :return: True if coordinates are equal, False otherwise
        """
        if isinstance(other, Point2D):
            return np.array_equal(self._coordinates, other.coordinates)
        return False

    def __ne__(self, other):
        """
        Check if two Point2D objects are not equal based on their coordinates.
        :param other: Another Point2D object
        :return: True if coordinates are not equal, False otherwise
        """
        return not self.__eq__(other)

    @property
    def coordinates(self) -> np.ndarray:
        """
        Get the coordinates of the point.
        :return: A copy of the coordinates as a numpy array
        """
        # Return a copy of the coordinates to prevent external modification
        return np.copy(self._coordinates)

    @property
    def x(self):
        """
        Get the x coordinate.
        :return: X coordinate
        """
        return self._coordinates[0]

    @property
    def y(self):
        """
        Get the y coordinate.
        :return: Y coordinate
        """
        return self._coordinates[1]

    def move(self, move: np.ndarray):
        """
        Move the point by a given vector.
        :param move: A 2D vector (dx, dy) to move the point
        """
        if move.shape != (2,):
            raise ValueError("Move vector must be 2D.")
        self._coordinates = np.add(self._coordinates, move)


    def __add__(self, other) -> np.ndarray:
        """
        Add two Point2D objects together.
        """
        if not isinstance(other, Point2D):
            raise TypeError("Only Point2D objects can be added.")

        if self._coordinates.shape != other.coordinates.shape:
            raise ValueError("Point2D objects must have the same shape.")

        return np.add(self._coordinates, other.coordinates)

    def __sub__(self, other: 'Point2D') -> np.ndarray:
        """
        Subtract two Point2D objects.
        :param other: Another Point2D object
        :return: Result of the subtraction as a numpy array
        """
        if not isinstance(other, Point2D):
            raise TypeError("Only Point2D objects can be subtracted.")

        if self._coordinates.shape != other.coordinates.shape:
            raise ValueError("Point2D objects must have the same shape.")

        return np.subtract(self._coordinates, other.coordinates)

    def distance(self, other: 'Point2D') -> float:
        """
        Calculate the distance to another point.
        :param other: Another Point2D object
        :return: Distance to the other point
        """
        return float(np.linalg.norm(self - other))

    def angle(self, other: 'Point2D') -> float:
        """
        Calculate the angle to another point in radians.
        :param other: Another Point2D object
        :return: Angle to the other point in radians
        """
        delta_y = other.y - self.y
        delta_x = other.x - self.x
        return np.arctan2(delta_y, delta_x)
