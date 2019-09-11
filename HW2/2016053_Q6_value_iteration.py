import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size, gamma):
        self.size = size
        self.gamma = gamma
        self.all_actions = {
            'left': np.array([0, -1]),
            'right': np.array([0, 1]),
            'up': np.array([-1, 0]),
            'down': np.array([1, 0])
        }

    def arg_max(self, all_awards):
        max_award = np.max(list(all_awards.values()))
        optimal_actions = []
        for each_action, award in all_awards.items():
            if award == max_award:
                optimal_actions.append(each_action)
        return optimal_actions

    def is_terminal(self, state):
        if (state == np.array([0, 0])).all() or (state == np.array([self.size-1, self.size-1])).all():
            return True
        else:
            return False

    def check_out_of_bounds(self, x, y):
        if x >= self.size or x < 0 or y >= self.size or y < 0:
            return True
        else:
            return False

    def value_iteration(self, threshold):
        value_map = np.zeros((self.size, self.size))
        iteration = 0
        while True:
            print('Iteration {0}'.format(iteration))
            print('Value Map')
            print(value_map)
            delta = 0
            for i in range(self.size):
                for j in range(self.size):
                    old_value_map = value_map.copy()
                    state = np.array([i, j])
                    all_awards = {}
                    for action_type, each_action in self.all_actions.items():
                        next_state, reward = self.take_step(state, each_action)
                        x, y = next_state
                        temp = 1 * (reward + (self.gamma * value_map[x, y]))
                        all_awards[action_type] = np.round(temp, decimals=2)
                    max_award = max(all_awards.values())
                    value_map[i, j] = max_award
                    delta = max(delta, np.abs(old_value_map[i, j] - max_award))
            iteration += 1
            if delta < threshold:
                break

        optimal_policy_map = np.empty((self.size, self.size), object)
        for i in range(self.size):
            for j in range(self.size):
                state = np.array([i, j])
                all_awards = {}
                for action_type, each_action in self.all_actions.items():
                    next_state, reward = self.take_step(state, each_action)
                    x, y = next_state
                    temp = 1 * (reward + (self.gamma * value_map[x, y]))
                    all_awards[action_type] = np.round(temp, decimals=2)
                optimal_policy_map[i, j] = self.arg_max(all_awards)

        print('Optimal Policy Map')
        print(optimal_policy_map)
        return value_map, optimal_policy_map

    def take_step(self, state, action):
        if self.is_terminal(state):
            next_state = state
            reward = 0
        else:
            next_state = state + action
            x, y = next_state
            if self.check_out_of_bounds(x, y):
                next_state = state
            reward = -1
        return next_state, reward

if __name__ == '__main__':
    grid_size = 4
    policy = {
        'left': 0.25,
        'right': 0.25,
        'up': 0.25,
        'down': 0.25
    }
    discount = 1
    gw = GridWorld(grid_size, discount)
    value_map, optimal_policy_map = gw.value_iteration(4e-6)