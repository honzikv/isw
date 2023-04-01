import argparse
import hashlib
import numpy as np
import random

from agent.qlearning_agent.qlearning_agent import QLearningAgent, QLearningConfig
from agent.qlearning_agent import serialization
from src.environment import FrozenLake
from src.play import Simulator


hash = lambda x: int(hashlib.md5(x).hexdigest(), 16)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output-file', type=str, help='Output file', required=False, default=None)

    args = argparser.parse_args()
    env = FrozenLake()
    agent = QLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.0,
            eps_decay_timesteps=7500,
            beta=0.2,
            gamma=.05,
            # candidate_q_table_update_frequency=10,
        )
    )

    # Set seed
    np.random.seed(0)
    random.seed(0)

    agent.train(n_episodes=50_000)

    if args.output_file is not None:
        serialization.save(file=args.output_file, qlearning_agent=agent)

    env.evaluate(agent=agent)
    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
