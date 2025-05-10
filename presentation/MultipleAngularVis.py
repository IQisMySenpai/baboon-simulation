import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

def angular_distribution(points, sigmas, resolution=1000):
    theta_grid = np.linspace(0, 2 * np.pi, resolution)
    probs = np.zeros_like(theta_grid)

    for mu, sigma in zip(points, sigmas):
        for k in range(-3, 4):
            probs += np.exp(-0.5 * ((theta_grid - mu + 2 * np.pi * k) / sigma) ** 2)

    probs /= np.trapezoid(probs, theta_grid)
    return theta_grid, probs

# --- Initialize multiple random points ---
num_points = 5
np.random.seed(0)

base_angles = np.random.uniform(0, 2 * np.pi, size=num_points)
base_distances = np.random.uniform(1, 3, size=num_points)

rotation_speeds = np.random.uniform(0.01, 0.05, size=num_points)  # radians per frame
zoom_amplitudes = np.random.uniform(0.2, 0.6, size=num_points)
zoom_frequencies = np.random.uniform(0.02, 0.05, size=num_points)

colors = cm.viridis(np.linspace(0, 1, num_points))

# --- Set up plots ---
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='polar')
ax2.set_title("Point Coordinates and Sigmas")

total_frames = 400

# --- Update function ---
def update(frame):
    ax1.clear()
    ax2.clear()

    # Continuous evolution of angles and radii
    angles = (base_angles + rotation_speeds * frame) % (2 * np.pi)
    distances = base_distances * (1 + zoom_amplitudes * np.sin(zoom_frequencies * frame * 2 * np.pi))

    sigmas = [1 / d for d in distances]
    theta_grid, probs = angular_distribution(angles, sigmas)
    probs /= np.trapz(probs, theta_grid)

    # Plot angular distribution
    ax1.plot(theta_grid, probs)
    ax1.set_title("Angular Probability Distribution")
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Probability Density")
    ax1.set_ylim(0, 0.8)
    ax1.grid(True)

    ax2.set_title("Point Coordinates and Sigmas")

    # Plot polar scatter
    ax2.set_ylim(0, 4.3)
    for theta, r, color in zip(angles, distances, colors):
        ax2.scatter(theta, r, color=color, s=50)

    return []

# Animate
ani = FuncAnimation(fig, update, frames=np.arange(0, total_frames), interval=50, blit=True)

plt.tight_layout()
ani.save("continuous_angular_distribution.gif", writer='pillow', fps=15)
