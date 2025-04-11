from map_entities.point2d import Point2D

class Baboon (Point2D):
    """
    A class representing a baboon in a 2D space.
    Inherits from Point2D to leverage its functionalities.
    """

    def __init__(self, x: float, y: float, color: str = "green"):
        """
        Initialize a Baboon object.
        :param x: X coordinate
        :param y: Y coordinate
        :param color: Color of the baboon (default is "green")
        """
        super().__init__(x, y, color)

    def __str__(self):
        return f"Baboon({self._coordinates[0]}, {self._coordinates[1]}, color={self.color})"

    def __repr__(self):
        return f"Baboon({self._coordinates[0]}, {self._coordinates[1]}, color={self.color})"
