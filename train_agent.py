import argparse
from agent import serialization

from queue import Queue

from agent.qlearning_agent import QLearningAgent, QLearningConfig
from src.environment import FrozenLake

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # argparser.add_argument('--n-episodes', type=int, help='Number of episodes', required=False, default=100_000)

    argparser.add_argument('--output-file', type=str, help='Output file', required=False, default='qlearning_agent.pkl')

    args = argparser.parse_args()

    env = FrozenLake()
    queue = Queue()

    agent = QLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.0,
            eps_decay_timesteps=2500,
            beta=0.2,
            gamma=.9,
            candidate_q_table_update_frequency=10,
        )
    )

    serialization.save(file=args.output_file, qlearning_agent=agent)
