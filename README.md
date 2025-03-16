# DynamicQVRP

This repository implements methods to solve the Dynamic Vehicle Routing Problem with emission Quota (DQVRP), a variant of the classic Vehicle Routing Problem that incorporates emissions considerations and dynamic customer requests.

## Overview

The Dynamic QVRP extends traditional routing problems by:

- Considering vehicle emissions alongside operational costs
- Handling dynamic customer requests that arrive over time
- Supporting multiple vehicle types with different emission profiles
- Allowing for capacity constraints and quota limitations
- Providing decision support for accepting/rejecting customer requests

## Key Features

- **Environment**: A custom Gymnasium environment (`DynamicQVRPEnv`) that simulates dynamic routing scenarios
- **Multiple Solution Methods**:
  - Reinforcement Learning (DQN, PPO)
  - Supervised Learning
  - Operations Research techniques (Simulated Annealing)
  - Greedy and baseline heuristics
- **Configurable Parameters**:
  - Vehicle fleet composition and capacities
  - Emission profiles
  - CO2 penalties
  - Demand patterns
  - Re-optimization strategies
- **Evaluation Tools**: Comprehensive metrics for comparing different solution approaches

## Installation

### Notable Prerequisites

- Python 3.8+
- PyTorch

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DynamicQVRP.git
cd DynamicQVRP
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv_DQVRP
source .venv_DQVRP/bin/activate  # On Windows: .venv_DQVRP\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

The main experiment function can be used to compare different solution methods:

```python
from experiments import experiment

# Configure environment parameters
env_configs = {
    "horizon": 50,
    "Q": 100,  # Emission quota
    "DoD": 0.5,  # Degree of Dynamism
    "vehicle_capacity": 25,
    "re_optimization": False,
    "emissions_KM": [0.1, 0.3],  # Emissions per km for each vehicle type
    "n_scenarios": 500
}

# Run experiment with default RL configuration
experiment(
    episodes=200,
    env_configs=env_configs,
    RL_hidden_layers=[1024, 1024, 1024]
)
```

### Training a Reinforcement Learning Agent

```python
from methods import RLAgent
from envs import DynamicQVRPEnv

# Create environment
env = DynamicQVRPEnv(
    horizon=50,
    Q=100,
    DoD=1.0,
    vehicle_capacity=20,
    costs_KM=[1, 1],
    emissions_KM=[0.1, 0.3],
    vehicle_assignment=True
)

# Create and train agent
agent = RLAgent(
    env,
    hidden_layers=[1024, 1024, 1024],
    learning_rate=0.0001,
    algo="custom_model",
    load_model = False,
)

# Run training
agent.train(episodes=1000)
```

### Using a Pre-trained Model

```python
from methods import RLAgent
from envs import DynamicQVRPEnv

env = DynamicQVRPEnv(
    horizon=100,
    Q=50,
    vehicle_capacity=20,
    emissions_KM=[0.1, 0.1, 0.3, 0.3]
)

agent = RLAgent(
   env,
   hidden_layers=[1024, 1024, 1024],
   algo="DQN_VRP4Q50_VA"
)

# Evaluate the agent
rewards, actions, infos = agent.run(episodes=100)
```

## Environment Configuration

The `DynamicQVRPEnv` class supports various configuration parameters:

| Parameter              | Description                       | Default |
| ---------------------- | --------------------------------- | ------- |
| `horizon`            | Number of time steps              | 50      |
| `Q`                  | Emission quota                    | 50      |
| `DoD`                | Degree of Dynamism                | 0.5     |
| `vehicle_capacity`   | Capacity per vehicle              | 15      |
| `emissions_KM`       | Emissions per km for each vehicle | [0.3]   |
| `re_optimization`    | Allow route re-optimization       | False   |
| `vehicle_assignment` | Allow vehicle selection           | False   |

## Project Structure

- `envs/`: Environment implementations
  - `envs.py`: Main QVRP environment
  - `assignment.py`: Assignment environment
- `methods/`: Solution methods
  - `ML/`: Machine learning methods (RL, SL)
  - `OR/`: Operations research methods
- `utils/`: Utility functions
- `data/`: Data files and scenarios
- `tests/`: Unit tests
- `results/`: Experiment results
- `experiments.py`: Main experiment runner
- `train.py`: Training script

## Examples

### Comparing Different Methods

```python
from experiments import experiment

# Configure environment
env_configs = {
    "horizon": 50,
    "Q": 100,
    "DoD": 0.5,
    "vehicle_capacity": 25,
    "emissions_KM": [0.1, 0.3]
}

# Run experiment comparing all methods
experiment(
    episodes=100,
    env_configs=env_configs
)
```

### Visualizing Routes

```python
from envs import DynamicQVRPEnv

env = DynamicQVRPEnv(
    horizon=50,
    Q=100,
    vehicle_capacity=20,
    emissions_KM=[0.1, 0.3]
)

env.reset()
# Run some steps
for _ in range(10):
    action = 1  # Accept request
    env.step(action)

# Render the current routes
env.render()
info
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
