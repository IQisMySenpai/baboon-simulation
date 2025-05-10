import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrow

# Parameters
K = 2  # Frequency parameter (adjust as needed)
T = 50  # Total time duration (seconds)
fps = 20  # Frames per second
total_frames = T * fps

# Create time array
t_values = np.linspace(0, T, total_frames)

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 8))

# Set axis limits
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(
    r'Time evolution of $\mu(t)$',
    fontsize=30,
)

# Initialize arrow (we'll update its properties rather than recreating it)
arrow = FancyArrow(0, 0, 0, 0, width=0.06, length_includes_head=True, color='r')
ax.add_patch(arrow)

# Add a dot at the origin
origin_dot = ax.scatter([0], [0], color='k', s=50)

# Add a time display text
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

# Add a path trace
trace, = ax.plot([], [], 'b-', alpha=0.5, lw=1)

# Store the trace points
trace_x, trace_y = [], []


def update(frame):
    t = t_values[frame] * 20

    # Calculate the vector components
    amplitude = 5 * np.sin(t/20)
    x = amplitude * np.sin(t/300)
    y = amplitude * np.cos(t/300)

    # Update arrow properties instead of clearing and recreating
    arrow.set_data(x=0, y=0, dx=x, dy=y)

    # Update the trace
    trace_x.append(x)
    trace_y.append(y)
    # trace.set_data(trace_x, trace_y)

    # Update time display
    # time_text.set_text(f"Time: {t:.2f}s\nVector: ({x:.2f}, {y:.2f})")

    return arrow, trace, time_text, origin_dot


# Create animation
ani = FuncAnimation(fig, update, frames=total_frames,
                    interval=1000/fps, blit=True)

# Uncomment to save as MP4 (requires ffmpeg)
ani.save('../outputs/wavy_drift_evolution.gif', writer='pillow', fps=fps)

plt.tight_layout()
plt.show()
