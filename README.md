# Snake AI with Reinforcement Learning

This project implements a Snake game with an AI agent that learns to play the game using reinforcement learning. The agent learns to navigate the snake to collect food while avoiding collisions with walls and its own body.

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/Szajsenberg/PBSnakeAI.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage

To train the AI agent, run the `train.py` script:

    ```
    python train.py
    ```

During training, the agent learns to play the game by interacting with the environment and updating its policy based on received rewards.

To test the trained agent, run the `test.py` script:

    ```
    python test.py
    ```

This script runs the trained agent in the game environment and prints the final score achieved.

## Components

- `game.py`: Implements the Snake game environment.
- `agent.py`: Defines the AI agent responsible for interacting with the game environment and learning.
- `model.py`: Contains the neural network model used by the AI agent.
- `plotter.py`: Provides functions for plotting game scores and performance metrics.


## Contributing

Oskar Walawender
