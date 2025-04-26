import numpy as np
import numpy.typing as npt


def get_angles(
    baboons: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the angles of each baboon with respect to every other baboon.

    Args:
        baboons_trajectory: Current baboons. Shape (t, n_baboons, 2)

    Returns:
        baboons_angles: Angles of each baboon with respect to every other
            baboon. Shape (n_baboons, n_baboons). baboons_angles[i, j] is the
            angle of baboon j with respect to baboon i.
    """
    baboon_differences = get_differences(baboons)
    baboon_angles = np.arctan2(
        baboon_differences[:, :, 1], baboon_differences[:, :, 0]
    )
    return baboon_angles


def get_differences(
    baboons: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the differences of each baboon with respect to every other baboon.

    Args:
        baboons: Baboon positions. Shape (n_baboons, 2)

    Returns:
        baboons_differences: Differences of each baboon with respect to every
            other baboon. Shape (n_baboons, n_baboons, 2).
            baboons_differences[i, j] is the vector from baboon i to baboon j.
    """
    return baboons[np.newaxis, :, :] - baboons[:, np.newaxis, :]


def get_distances(
    baboons: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the distances of each baboon with respect to every other baboon
    Args:
        baboons: Baboon positions. Shape (n_baboons, 2)
    Returns:
        baboons_distances: Distances of each baboon with respect to every
            other baboon. Shape (n_baboons, n_baboons).
            baboons_distances[i, j] is the distance from baboon i to baboon j.
    """
    baboon_differences = get_differences(baboons)
    baboon_distances = np.linalg.norm(baboon_differences, axis=2)
    return baboon_distances