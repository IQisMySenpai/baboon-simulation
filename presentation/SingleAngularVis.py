import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def angular_distribution(points, sigmas, resolution=1000):
    theta_grid = np.linspace(0, 2 * np.pi, resolution)
    probs = np.zeros_like(theta_grid)

    for mu, sigma in zip(points, sigmas):
        for k in range(-3, 4):
            probs += np.exp(-0.5 * ((theta_grid - mu + 2 * np.pi * k) / sigma) ** 2)

    # Normalize the distribution
    probs /= np.trapezoid(probs, theta_grid)

    return theta_grid, probs


# Define initial polar points (angle + distance)
angles = [0.0]
distances = [3]  # These could inversely relate to sigma
sigmas = [1 / d for d in distances]  # Smaller sigma for larger distance

# --- Set up plots for animation ---
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='polar')

# Plot 1: Angular distribution (static)
theta_grid, probs = angular_distribution(angles, sigmas)
probs /= np.trapezoid(probs, theta_grid)
ax1.plot(theta_grid, probs)
ax1.set_title("Angular Probability Distribution")
ax1.set_xlabel("Angle (radians)")
ax1.set_ylabel("Probability Density")
ax1.grid(True)

ax2.set_title("Point Coordinates and Sigmas")

# Colors for points
colors = ['red']

# Scatter plot for polar coordinates (empty initialization)
scatter = ax2.scatter([], [], color=[], s=50)

total_frames = 400  # 200 for rotation + 200 for zoom effect

# Update function for animation
def update(frame):
    ax1.clear()
    ax2.clear()

    # Determine the phase: 0-99 for rotation, 100-199 for zoom effect
    if frame < 200:
        # Rotation phase
        updated_angles = [angle + 2 * np.pi / 200 * frame for angle in angles]
        updated_distances = distances
    else:
        # Zoom effect phase
        updated_angles = angles  # Keep the angle constant
        zoom_factor = 1 - 0.5 * np.sin((frame - 200) * np.pi / 200)  # Sine wave zoom effect
        updated_distances = [d * zoom_factor for d in distances]

    # Recalculate and normalize probability distribution
    updated_sigmas = [1 / d for d in updated_distances]
    theta_grid, updated_probs = angular_distribution(updated_angles, updated_sigmas)
    updated_probs /= np.trapz(updated_probs, theta_grid)

    # --- Update angular distribution plot ---
    ax1.plot(theta_grid, updated_probs)
    ax1.set_title("Angular Probability Distribution")
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Probability Density")
    ax1.grid(True)

    ax2.set_ylim(0, 3.3)  # Fixed zoom, adjust if needed
    ax1.set_ylim(0, 1.3)  # Fixed y-limits for the probability distribution

    # --- Update polar plot ---
    ax2.set_title("Point Coordinates and Sigmas")
    for i, (theta, r, sigma) in enumerate(zip(updated_angles, updated_distances, updated_sigmas)):
        ax2.scatter(theta, r, color=colors[i], s=50)

    return []

# Set up the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, total_frames), interval=50, blit=True)

plt.tight_layout()
ani.save("angular_distribution_animation.gif", writer='pillow', fps=20)
