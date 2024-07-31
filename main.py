# File: utils.py
"""Utility functions for the training process."""

import matplotlib.pyplot as plt


def plot_metrics(metrics):
    """Plot and save the training metrics."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Training Metrics')

    for i, (key, values) in enumerate(metrics.items()):
        row, col = divmod(i, 2)
        axs[row, col].plot(values)
        axs[row, col].set_title(key.replace('_', ' ').title())
        axs[row, col].set_xlabel('Iterations')
        axs[row, col].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# File: main.py
"""Main script to run the training process."""

import ray
from ray.tune.registry import register_env
from environment import create_env
from training import setup_training, train
from utils import plot_metrics


def main():
    """Main function to run the training process."""
    NUM_AGENTS = 5
    NUM_ITERATIONS = 200
    CHECKPOINT_FREQ = 20

    ray.init(num_gpus=1)

    register_env("cleanup_game", lambda config: create_env(NUM_AGENTS))

    trainer = setup_training(NUM_AGENTS)
    metrics = train(trainer, NUM_ITERATIONS, CHECKPOINT_FREQ)
    plot_metrics(metrics)

    ray.shutdown()


if __name__ == "__main__":
    main()
