# Baboon Simulation

This project simulates the movement of baboons using Python. It uses multiple visualization techniques and can be run with a pre-configured environment.

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
