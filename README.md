# Baboon Simulation

This project simulates the movement of baboons using Python. It uses multiple visualization techniques and can be run with a pre-configured environment. Read the [Setup](#setup) section to get started and [Understanding the Simulation architecture](#understanging-the-simulation-architecture) to learn how to create new strategies.

## Understanging the simulation architecture

First of all, **read `src/simulation_types/documentation.py`**: it contains the explanation of the **objects** in the simulation and the **SDEs** used to simulate the baboon movement. The important part here is the **drift** function, which takes the past trajectories of all baboons and returns the step vector that each baboon is going to take in the next time-step. Read the `src/simulation_types/documentation.py` file to understand the details. The full simulation is explained there and it is not a long explanation.

Therefore, in order to create a new strategy (i.e. rules) for the baboons to follow, you need to implement a new drift function. The workflow is intended to be as follows:

1. Modify the "SIMULATION PARAMETERS" in `src/main.py` (e.g. number of baboons, initial position of baboons, total number of timesteps, etc.)

2. Create a drift function in `src/strategies/drift.py` by re-using the existing ones (or not) and adding your own logic. You can use/add any other helper functions in `src/utils/baboons.py` if needed. The idea is that drift functions are configurable with parameters, you can follow the same pattern as I used for `src.strategies.drift.only_angles_drift_function` to achieve this.

3. Then, import your drift function and use it in `src/main.py` to run the simulation with your new strategy. Add any "SIMULATION PARAMETERS" that you need for your own drift function.

4. Finally, run the `src/main.py` script and visualize the results to adjust your parameters/strategy.

(Pablo)

## Setup
Follow these steps to set up and run the Baboon Simulation.

### 1. Create a Conda Environment with Python 3.9.7
To start, create a new Conda environment with Python version 3.9.7:

```bash
conda create --name baboon-simulation python=3.9.7
```

Once the environment is created, activate it with the following command:

```bash
conda activate baboon-simulation
```

### 2. Install Dependencies

Install the required dependencies using the `requirements.txt` file. First, make sure you have the `requirements.txt` file in your project root directory, and then run:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries for running the simulation, including `matplotlib`, `numpy`, and other dependencies required for the project.

### 4. Set the Source Directory

Ensure that the source directory is set to `src`. You can set this in your IDE or from the terminal by modifying your `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)/src
```

This tells Python to treat the `src` folder as the source root for the project.

### 5. Run the Simulation

To run the simulation, use the following command:

```bash
python src/main.py
```

This will start the simulation and execute the code defined in `main.py`.
