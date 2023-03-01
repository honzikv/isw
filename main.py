from agent.q_learning_agent import QLearningAgent
from src.agent import RandomAgent, BalancingAgent
from src.environment import Connect4, TicTacToe, FrozenLake, CartPole, MountainCar, LectureExample, MultiArmedBandit
from src.play import Simulator
from collections import Counter

if __name__ == '__main__':
    env = FrozenLake()
    state_shape = env.observation_shape

    agent = QLearningAgent(
        environment=env,
        eps_init=1.0,
        eps_final=0.15,
        eps_decay_timesteps=1000
    )

    agent.train(n_episodes=10000)

    print('')
    # state, _, valid_actions, is_terminal = env.set_to_initial_state()
    # total_reward = .0
    # while not is_terminal:
    #     random_action = agent.best_action()
    #
    # sim = Simulator(env, agent, fps=0)
    # sim.show()
    # sim.run()
