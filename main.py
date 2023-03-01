from agent.qlearning_agent import QLearningAgent, QLearningConfig
from src.agent import RandomAgent, BalancingAgent
from src.environment import Connect4, TicTacToe, FrozenLake, CartPole, MountainCar, LectureExample, MultiArmedBandit
from src.play import Simulator
from collections import Counter

if __name__ == '__main__':
    env = FrozenLake()
    state_shape = env.observation_shape

    agent = QLearningAgent(
        env=env,
        config=QLearningConfig(
            eps_init=1.0,
            eps_final=0.00,
            eps_decay_timesteps=500_000,
            eps_decay_type='linear',
            use_candidate_q_table=False,
            use_experience_buffer=False,
            beta=0.9,
            gamma=0.999
        )
    )

    agent.train(n_episodes=1_000_000)
    agent.eval(n_episodes=10_000)

    # state, _, valid_actions, is_terminal = env.set_to_initial_state()
    # total_reward = .0
    # while not is_terminal:
    #     random_action = agent.best_action()
    #

    sim = Simulator(env, agent, fps=0)
    sim.show()
    sim.run()
