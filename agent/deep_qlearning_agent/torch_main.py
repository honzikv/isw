import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

import platform
import numpy as np
import wandb

from agent.deep_qlearning_agent.deep_qlearning_agent import DeepQLearningNet, ReplayMemory, DeepQLearningAgent, \
    QLearningConfig
from src.environment import CartPole

if __name__ == '__main__':
    env = CartPole()
    device = torch.device('cpu')
    run = wandb.init(project='deep-qlearning')

    run.define_metric('episode_reward', step_metric='episode')
    run.define_metric('episode_length', step_metric='episode')
    run.define_metric('eps', step_metric='episode')

    agent = DeepQLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.05,
            eps_decay_timesteps=10_000,
            beta=0.15,
            gamma=.99,
            replay_memory_size=10_000,
            batch_size=128,
            lr=1e-5,
            wandb_logging=True,
        ),
    )

    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(agent.policy_net.parameters(), lr=agent.config.lr, amsgrad=True)
    agent.train(loss_fn, optimizer, n_episodes=10_000)


