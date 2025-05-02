from datetime import datetime, timedelta
import numpy as np
from simulation.sim_output import SimOutput
import numpy.typing as npt


class GPSCSVExporter(SimOutput):
    def __init__(
        self,
        start_time: datetime = datetime(2012, 8, 1, 2, 59, 57),
        time_step_seconds: int = 1,
        base_lat: float = 0.3509,
        base_long: float = 36.9229,
        scale: float = 1e-4,
    ):
        """
        Initialize the GPS CSV exporter.

        Args:
            start_time: The starting datetime for the simulation.
            time_step_seconds: Seconds between each timestep.
            base_lat: Latitude to map (0,0) to.
            base_long: Longitude to map (0,0) to.
            scale: Scaling factor from simulation units to degrees.
        """
        self.start_time = start_time
        self.time_step = timedelta(seconds=time_step_seconds)
        self.base_lat = base_lat
        self.base_long = base_long
        self.scale = scale

    def save(
            self,
            baboons_trajectory: npt.NDArray[np.float64],
            filename: str
    ):
        """
        Save the baboon trajectory as a timestamped GPS CSV file.

        Args:
            baboons_trajectory: Full trajectory (steps, n_baboons, 2).
            filename: Path to the output CSV file (without extension).
        """
        rows = []

        n_steps, n_baboons, _ = baboons_trajectory.shape
        for t in range(n_steps):
            timestamp = self.start_time + t * self.time_step
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            for i in range(n_baboons):
                x, y = baboons_trajectory[t, i]
                lon = self.base_long + x * self.scale
                lat = self.base_lat + y * self.scale
                identifier = f"{i + 1}"
                rows.append((
                    identifier,
                    timestamp_str,
                    f"{lon:.7f}",
                    f"{lat:.7f}",
                ))

        rows.sort(key=lambda r: (r[0], r[1]))

        with open(f"{filename}.csv", "w", newline="") as f:
            f.write("timestamp,location-long,location-lat,individual-local-identifier\n")
            for identifier, timestamp, lon, lat in rows:
                f.write(f'{timestamp},{lon},{lat},"{identifier}"\n')

        print(f"GPS CSV saved to {filename}.csv")
