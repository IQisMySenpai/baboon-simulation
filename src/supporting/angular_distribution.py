import numpy as np

def angular_distribution(angles, sigmas, resolution=1000):
    """
    Calculates a smooth angular probability distribution over [0, 2π),
    based on a set of directional "preferences" modeled as wrapped Gaussians.

    Parameters:
    :param angles: list or array of mean angles (in radians) representing the preferred directions of other entities.
    :param sigmas: list or array of standard deviations for each angle, representing directional uncertainty.
    :param resolution: number of points in the angle grid for evaluating the distribution (default: 1000)

    :return: tuple (theta_grid, probs) array of angle values in [0, 2π) used for evaluating the probability distribution,
    normalized probability density values corresponding to each angle in theta_grid.
    """

    # Create a grid of angles from 0 to 2π for evaluating the distribution
    theta_grid = np.linspace(0, 2 * np.pi, resolution)

    # Initialize the probability array with zeros
    probs = np.zeros_like(theta_grid)

    # For each directional preference (mu) and uncertainty (sigma)
    for mu, sigma in zip(angles, sigmas):
        # Wrap the Gaussian distribution to account for the circular nature of angles
        # Since the normal distribution isn't periodic, we approximate the wrapped effect
        # by summing over multiple shifted copies (3 in each direction is a good approximation)
        for k in range(-3, 4):
            # Add the contribution of this wrapped Gaussian to the total distribution
            wrapped_shift = theta_grid - mu + 2*np.pi*k
            probs += np.exp(-0.5 * (wrapped_shift / sigma)**2)

    # Normalize the probability distribution so that the total area under the curve is 1
    # This ensures it's a valid probability density function
    probs /= np.trapz(probs, theta_grid)

    return theta_grid, probs


def sample_from_distribution(theta_grid, probs, n_samples=1):
    """
    Samples angle values from a given angular probability distribution.

    :param theta_grid: array of angles [0, 2π) corresponding to the probability density values.
    :param  probs: array of normalized probability values (same size as theta_grid).
    :param n_samples: number of samples to draw from the distribution.

    :return: array of sampled angle values from the input distribution.
    """

    # Compute the cumulative distribution function (CDF) from the probability density
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]  # Normalize the CDF to go from 0 to 1

    # Generate uniform random samples and use the CDF to map them to angle values
    random_vals = np.random.rand(n_samples)
    sampled_thetas = np.interp(random_vals, cdf, theta_grid)

    return sampled_thetas
