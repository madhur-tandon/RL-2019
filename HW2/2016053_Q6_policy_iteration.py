import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size, gamma, policy):
        self.size = size
        self.gamma = gamma
        self.all_actions = {
            'left': np.array([0, -1]),
            'right': np.array([0, 1]),
            'up': np.array([-1, 0]),
            'down': np.array([1, 0])
        }
        self.policy = policy
        self.policy_probs_map = np.array([[self.policy.copy() for i in range(self.size)] for j in range(self.size)])

    def arg_max(self, all_awards, i, j):
        max_award = np.max(list(all_awards.values()))
        optimal_actions = []
        c = 0
        for each_action, award in all_awards.items():
            if award == max_award:
                optimal_actions.append(each_action)
                c += 1
        for each_action, award in all_awards.items():
            if award == max_award:
                self.policy_probs_map[i][j][each_action] = 1/c
            else:
                self.policy_probs_map[i][j][each_action] = 0
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

    def policy_evaluation(self, threshold):
        value_map = np.zeros((self.size, self.size))
        while True:
            delta = 0
            for i in range(self.size):
                for j in range(self.size):
                    v = 0
                    state = np.array([i, j])
                    for action_type, each_action in self.all_actions.items():
                        next_state, reward = self.take_step(state, each_action)
                        x, y = next_state
                        v += self.policy_probs_map[i][j][action_type] * 1 * (reward + (self.gamma * value_map[x, y]))
                    delta = max(delta, np.abs(v - value_map[i, j]))
                    value_map[i, j] = v
            if delta < threshold:
                break

        return value_map

    def policy_improvement(self, threshold):
        policy_map = np.empty((self.size, self.size), object)
        iteration = 0
        
        while True:
            value_map = self.policy_evaluation(threshold)
            print('Iteration {0}'.format(iteration))
            print('Value Map')
            print(value_map)
            print('Policy Map')
            print(policy_map)
            policy_stable = True
            for i in range(self.size):
                for j in range(self.size):
                    old_policy_map = policy_map.copy()
                    state = np.array([i, j])
                    all_awards = {}
                    for action_type, each_action in self.all_actions.items():
                        next_state, reward = self.take_step(state, each_action)
                        x, y = next_state
                        temp = 1 * (reward + (self.gamma * value_map[x, y]))
                        all_awards[action_type] = np.round(temp, decimals=2)
                    policy_map[i, j] = self.arg_max(all_awards, i, j)
                    if old_policy_map[i,j] != policy_map[i, j]:
                        policy_stable = False
            iteration += 1
            if policy_stable:
                return np.round(value_map, decimals=1), policy_map

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
    gw = GridWorld(grid_size, discount, policy)
    value_map, optimal_policy_map = gw.policy_improvement(1e-5)