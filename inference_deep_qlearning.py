import argparse
import pytorch_lightning as pl
from pathlib import Path

from agent.deep_qlearning_agent.deep_qlearning_agent import DeepQLearningAgent
from agent.deep_qlearning_agent.deep_qlearning_module import DeepQModuleLightning, DeepQLearningConfig
from src.environment import CartPole, Environment
from src.play import Simulator


def main(args):
    env = CartPole()

    if not args.weights_path.exists():
        print(f'Weights file {args.weights_path} does not exist')
        exit(1)

    deepq_model = DeepQModuleLightning.load_from_checkpoint(args.weights_path, env=env, config=DeepQLearningConfig(),
                                                            device=None)

    pl.seed_everything(42069)

    net = deepq_model.target_net
    agent = DeepQLearningAgent.from_pretrained(
        env=env,
        net=net,
    )

    simulator = Simulator(env, agent, fps=10)
    simulator.show()
    simulator.run()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--weights-path', type=Path, required=True, default='weights.ckpt')

    args = argparser.parse_args()

    main(args)
