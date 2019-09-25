import numpy as np
import matplotlib.pyplot as plt

grid = np.full((4, 12), -1)
grid[3, 1:11] = -100

start_state = [3, 0]
end_state = [3, 11]

action_dict = {
    0: 'up',
    1: 'left',
    2: 'right',
    3: 'down'
}
all_actions = [0, 1, 2, 3]

epsilon = 0.1
alpha = 0.5
gamma = 1

def take_action(state, action):
    i, j = state
    if action_dict[action] == 'up':
        if (i - 1) >= 0:
            next_state = [i - 1, j]
        else:
            next_state = [0, j]
    elif action_dict[action] == 'left':
        if (j - 1) >= 0:
            next_state = [i, j - 1]
        else:
            next_state = [i, 0]
    elif action_dict[action] == 'right':
        if (j + 1) < 12:
            next_state = [i, j + 1]
        else:
            next_state = [i, 11]
    elif action_dict[action] == 'down':
        if (i + 1) < 4:
            next_state = [i + 1, j]
        else:
            next_state = [3, j]

    p, q = next_state
    reward = grid[p, q]

    if reward == -100:
        next_state = start_state

    if state == end_state:
        next_state == end_state
    
    return next_state, reward

def get_action(state, Q_value):
    i, j = state
    all_action_values = Q_value[i, j, :]
    max_action_value = np.amax(all_action_values)

    all_argmax_actions = [i for i, a in enumerate(all_action_values) if a == max_action_value]
    epsilon_toss = np.random.binomial(1, epsilon)
    if epsilon_toss == 1:
        return np.random.choice(all_actions)
    else:
        return np.random.choice(all_argmax_actions)

def sarsa(Q_value):
    current_state = start_state
    current_action = get_action(current_state, Q_value)
    total_reward = 0
    while current_state != end_state:
        next_state, reward = take_action(current_state, current_action)
        next_action = get_action(next_state, Q_value)
        total_reward += reward
        i, j = current_state
        p, q = next_state
        Q_value[i, j, current_action] += alpha * (reward + (gamma * Q_value[p, q, next_action]) - Q_value[i, j, current_action])
        current_state = next_state
        current_action = next_action

    return total_reward

def Q_learning(Q_value):
    current_state = start_state
    total_reward = 0
    while current_state != end_state:
        current_action = get_action(current_state, Q_value)
        next_state, reward = take_action(current_state, current_action)
        total_reward += reward
        i, j = current_state
        p, q = next_state
        Q_value[i, j , current_action] += alpha * (reward + (gamma * np.amax(Q_value[p, q, :])) - Q_value[i, j, current_action])
        current_state = next_state
    return total_reward


if __name__ == '__main__':
    num_runs = 300
    num_episodes = 500

    all_awards = {
        'sarsa': np.zeros(num_episodes),
        'q-learning': np.zeros(num_episodes)
    }

    for each_run in range(num_runs):
        Q_function_SARSA = np.zeros((4, 12, len(all_actions)))
        Q_function_Q_learning = np.zeros((4, 12, len(all_actions)))
        for each_episode in range(num_episodes):
            all_awards['sarsa'][each_episode] += sarsa(Q_function_SARSA)
            all_awards['q-learning'][each_episode] += Q_learning(Q_function_Q_learning)

    all_awards['sarsa'] /= num_runs
    all_awards['q-learning'] /= num_runs

    plt.figure()
    plt.plot(all_awards['sarsa'], label='Sarsa')
    plt.plot(all_awards['q-learning'], label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, -15])
    plt.legend()
    plt.show()