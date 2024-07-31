# File: training.py
"""Training setup and execution for PPO on Melting Pot Clean-up game."""

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from model import CustomCNNModel
from environment import create_env
from utils import plot_metrics


def setup_training(num_agents):
    """Set up the training configuration."""
    ray.init(num_gpus=1)

    ModelCatalog.register_custom_model("custom_cnn_model", CustomCNNModel)

    policies = {f"player_{i}" for i in range(num_agents)}

    config = (
        PPO.get_default_config()
        .environment("cleanup_game")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .training(
            model={"custom_model": "custom_cnn_model"},
            vf_loss_coeff=0.005,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .framework("torch")
        .resources(num_gpus=1)
        .rollouts(num_env_runners=2)
    )

    return PPO(config=config)


def train(trainer, num_iterations, checkpoint_freq):
    """Execute the training loop."""
    metrics = {
        "episode_reward_mean": [],
        "episode_len_mean": [],
        "waste_cleaned": [],
        "apples_consumed": [],
    }

    for i in range(num_iterations):
        result = trainer.train()

        for key in metrics:
            if key in result:
                metrics[key].append(result[key])
            elif key in result.get("custom_metrics", {}):
                metrics[key].append(result["custom_metrics"][key])
            else:
                metrics[key].append(0)

        print(f"Iteration {i + 1}/{num_iterations}, "
              f"episode_reward_mean: {result['episode_reward_mean']:.2f}, "
              f"episode_len_mean: {result['episode_len_mean']:.2f}, "
              f"waste_cleaned: {metrics['waste_cleaned'][-1]:.2f}, "
              f"apples_consumed: {metrics['apples_consumed'][-1]:.2f}")

        if (i + 1) % checkpoint_freq == 0:
            checkpoint = trainer.save()
            print(f"Checkpoint saved at {checkpoint}")

    final_checkpoint = trainer.save()
    print(f"Final model saved at {final_checkpoint}")

    return metrics
