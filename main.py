from agent.qlearning_agent import QLearningAgent, QLearningConfig
from src.agent import RandomAgent, BalancingAgent
from src.environment import Connect4, TicTacToe, FrozenLake, CartPole, MountainCar, LectureExample, MultiArmedBandit
from src.play import Simulator
from collections import Counter

if __name__ == '__main__':
    env = FrozenLake()

    agent = QLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.0,
            eps_decay_timesteps=4_000,
            beta=0.1,
            gamma=.78,
        )
    )

    agent.train(n_episodes=50_000)
    agent.eval(n_episodes=10_000)

    env.evaluate(agent)

    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
