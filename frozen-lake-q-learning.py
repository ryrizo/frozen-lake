import numpy as np
import gym
import random

# Hyperparameters
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate
start_epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

def main():
    env = gym.make("FrozenLake-v0")
    q_table = _initialize_qtable(env.observation_space.n, env.action_space.n)
    learned_q = learn_q_values(env, q_table, start_epsilon)

    env.reset()
    play_frozen_lake(env, learned_q)
    env.close()


def learn_q_values(env, q_table, epsilon):
    rewards = list()
    for episode in range(total_episodes):

        state = env.reset()
        cumulative_episode_reward = 0

        for step in range(max_steps):

            action = _choose_action(q_table, epsilon, state, env.action_space)

            (new_state, reward, done, probabilities) = env.step(action)

            q_table[state, action] = _get_updated_q_value(q_table, state, new_state, action, reward)

            state = new_state

            cumulative_episode_reward += reward

            if done:
                break

        rewards.append(cumulative_episode_reward)
        epsilon = _decay_epsilon(episode)

    return q_table


def play_frozen_lake(env, q_table):
    for episode in range(5):
        state = env.reset()
        for step in range(max_steps):
            action = np.argmax(q_table[state])
            (new_state, reward, done, prob) = env.step(action)
            state = new_state

            if done:
                print("Number of steps {}", step)
                env.render()
                break

def _initialize_qtable(number_of_states, number_of_actions):
    return np.zeros((number_of_states, number_of_actions))


def _choose_action(q_table, epsilon, state, action_space):
    return action_space.sample if _should_explore(epsilon) else np.argmax(q_table[state, :])


def _should_explore(epsilon):
    return random.uniform(0, 1) < epsilon


def _get_updated_q_value(q_table, state, new_state, action, reward):
    current_q = q_table[state][action]
    return current_q + learning_rate * (reward + gamma * q_table[new_state, :].max() - current_q)


def _decay_epsilon(episode):
    return min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)


if __name__ == '__main__':
    main()
