import numpy as np

def angular_distribution(points, sigmas, resolution=1000):
    theta_grid = np.linspace(0, 2 * np.pi, resolution)
    probs = np.zeros_like(theta_grid)

    for mu, sigma in zip(points, sigmas):
        for k in range(-3, 4):  # wrap approximation
            probs += np.exp(-0.5 * ((theta_grid - mu + 2*np.pi*k)/sigma)**2)

    probs /= np.trapz(probs, theta_grid)  # normalize
    return theta_grid, probs

def sample_from_distribution(theta_grid, probs, n_samples=1):
    cdf = np.cumsum(probs)
    cdf /= cdf[-1]  # normalize CDF

    # Uniform samples and interpolation
    random_vals = np.random.rand(n_samples)
    sampled_thetas = np.interp(random_vals, cdf, theta_grid)
    return sampled_thetas