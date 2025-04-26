from typing import Optional
import numpy.typing as npt
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch
from simulation_types.documentation import (
    DriftType, DriftDiffusionWithStateType, DiffusionType,
)


class Simulator:
    """
    Simulator class for baboon movement simulation.

    This class is responsible for simulating the movement of baboons
    based on a given Stochastic Differential Equation (SDE) model.
    In particular, it implements the Euler scheme for the SDE.
    Upon initialization, it takes the parameters for the Euler scheme,
    including the drift and diffusion functions.

    See simulation_types/documentation.py for more details on the object
    meanings and SDE formulation of the problem.

    Parameters:
        total_time_steps: Total number of simulation steps.
        initial_baboons: Initial positions of baboons. Shape: (n_baboons, 2).
        dt: Time step size. Default is 1.
        seed: Random seed for reproducibility. Default is 0.
        drift_diffusion_with_state: Callable function that takes the
            baboons_trajectory, random generator, and state as input
            and returns the drift, diffusion, and next state.
            Use this instead of drift and diffusion parameters.
    Parameters set after "run" method is called:
        baboons_trajectory_ (np.ndarray): Full trajectory of baboons.
            Shape: (total_steps + 1, n_baboons, 2).
    """
    total_steps: int
    initial_baboons: npt.NDArray[np.float64]
    dt: float = 1
    seed: int = 0
    drift: Optional[DriftType] = None
    diffusion: Optional[DiffusionType] = None

    # Parameters set after "run" method is called
    baboons_trajectory_: npt.NDArray[np.float64]

    def __init__(
        self,
        total_time_steps: int,
        initial_baboons: npt.NDArray[np.float64],
        dt: float = 1,
        drift: Optional[DriftType] = None,
        diffusion: Optional[DiffusionType] = None,
        seed: int = 0,
        drift_diffusion_with_state: Optional[DriftDiffusionWithStateType] = (
            None
        ),
    ):
        """
        Initialize the simulator with a given number of simulation steps.
        """
        self.total_steps = total_time_steps
        self.initial_baboons = initial_baboons
        self.dt = dt
        self.drift = drift
        self.diffusion = diffusion
        self.seed = seed

        assert initial_baboons.shape[1] == 2, (
            "Initial baboons must have shape (n_baboons, 2)"
        )

        if drift_diffusion_with_state is not None:
            assert drift is None, (
                "drift_diffusion_with_states is not None, "
                "but drift also provided."
            )
            assert diffusion is None, (
                "drift_diffusion_with_states is not None, "
                "but diffusion also provided."
            )
            drift, diffusion, _ = drift_diffusion_with_state(
                initial_baboons[np.newaxis, :, :],
                np.random.default_rng(seed),
                None,
            )
            assert diffusion.shape[0] == initial_baboons.shape[0], (
                "Diffusion function must return a diffusion term of shape "
                "(n_baboons, J) where J is the number of Wiener processes."
            )
            assert drift.shape[0] == initial_baboons.shape[0], (
                "Drift function must return a drift term of shape "
                "(n_baboons, 2)."
            )
            self.bm_dim = (diffusion.shape[0], diffusion.shape[2])
        else:  # I would warn that this is deprecated
            print(
                "Warning: using drift and diffusion functions without state "
                "may cause problems. Use drift_diffusion_with_state instead."
            )
            if self.drift is None:
                self.drift = lambda *_: np.zeros_like(initial_baboons)
            if self.diffusion is None:
                self.diffusion = lambda *_: np.zeros((
                    initial_baboons.shape[0], 1,
                ))
                self.bm_dim = 0  # No diffusion term
            else:
                diffusion = diffusion(
                    initial_baboons[np.newaxis, :, :],
                    np.random.default_rng(seed),
                    None,
                )
                self.bm_dim = (diffusion.shape[0], diffusion.shape[2])

        self.drift_diffusion_with_state = drift_diffusion_with_state

    def run(
        self,
        seed: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Run the Euler scheme and return full trajectories.

        This method performs the simulation of baboon movements using the
        Euler scheme based on the provided drift and diffusion functions.
        After the simulation, it returns the full trajectory of baboons.
        self.baboons_trajectory_ is also updated with the full trajectory.

        Args:
            seed (int, optional): Random seed for reproducibility. If None,
                the self.seed is used. Only specify this seed if you want to
                override the use of self.seed. In any case, self.seed is not
                modified.

        Returns:
            baboons_trajectory: Full trajectory of baboons. Shape:
                (total_steps + 1, n_baboons, 2)
        """
        # Use random generator for reproducibility
        # recommended for numpy rather thatn using random.seed
        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed)  # Create a random generator

        baboons_trajectory = np.zeros(
            (self.total_steps + 1, self.initial_baboons.shape[0], 2),
            dtype=np.float64,
        )
        baboons_trajectory[0] = self.initial_baboons

        if self.drift_diffusion_with_state is not None:
            state = None  # Initialize state to None. It will be updated later.
            for i in tqdm(range(self.total_steps), desc="Euler iterations"):
                drift, diffusion, state = self.drift_diffusion_with_state(
                    baboons_trajectory[:i + 1], rng, state,
                )
                bm_increment = rng.normal(
                    0, np.sqrt(self.dt), size=self.bm_dim,
                )
                diffusion_term = np.einsum(
                    "mij,mj->mi",
                    diffusion,
                    bm_increment,
                )
                baboons_trajectory[i + 1] = (
                    baboons_trajectory[i]
                    + drift * self.dt
                    + diffusion_term
                )
        else:
            for i in tqdm(range(self.total_steps), desc="Euler iterations"):
                drift, state = self.drift(baboons_trajectory[:i + 1], rng)
                if self.bm_dim != 0:
                    bm_increment = rng.normal(
                        0, np.sqrt(self.dt), size=self.bm_dim,
                    )
                    diffusion_term = np.einsum(
                        "mij,mj->mi",
                        self.diffusion(baboons_trajectory[:i + 1], rng),
                        bm_increment,
                    )
                else:
                    diffusion_term = 0
                baboons_trajectory[i + 1] = (
                    baboons_trajectory[i]
                    + drift * self.dt
                    + diffusion_term
                )

        self.baboons_trajectory_ = baboons_trajectory
        return baboons_trajectory
