import argparse
import hashlib
from pathlib import Path

import numpy as np
import random

from agent.qlearning_agent import serialization
from agent.qlearning_agent.qlearning_agent import QLearningAgent, QLearningConfig
from src.environment import FrozenLake
from src.play import Simulator

hash = lambda x: int(hashlib.md5(x).hexdigest(), 16)

if __name__ == '__main__':
    env = FrozenLake()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model-path', type=Path, required=True)
    args = argparser.parse_args()

    agent = serialization.load(file=args.model_path, agent=QLearningAgent(
        env=env,
        config=QLearningConfig(),  # dummy config this is overriden
    ))

    env.evaluate(agent=agent)
    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
