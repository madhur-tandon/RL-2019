import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size, A, A_prime, B, B_prime, gamma):
        self.size = size
        self.A = A
        self.A_prime = A_prime
        self.B = B
        self.B_prime = B_prime
        self.gamma = gamma
        self.all_actions = {
            'left': np.array([0, -1]),
            'right': np.array([0, 1]),
            'up': np.array([-1, 0]),
            'down': np.array([1, 0])
        }

    def check_out_of_bounds(self, x, y):
        if x >= self.size or x < 0 or y >= self.size or y < 0:
            return True
        else:
            return False

    def take_step(self, state, action):
        if (state == self.A).all():
            next_state = self.A_prime
            reward = 10
        elif (state == self.B).all():
            next_state = self.B_prime
            reward = 5
        else:
            next_state = state + action
            x, y = next_state
            if self.check_out_of_bounds(x, y):
                next_state = state
                reward = -1
            else:
                reward = 0
        
        return next_state, reward

    def find_value_function_given_policy_by_matrix(self, policy):
        num_states = self.size * self.size
        matrix_A = np.identity(num_states)
        matrix_B = np.zeros((num_states))
        for i in range(self.size):
            for j in range(self.size):
                state = np.array([i, j])
                state_index = j + (self.size * i)
                for action_type, each_action in self.all_actions.items():
                    next_state, reward = self.take_step(state, each_action)
                    x, y = next_state
                    new_state_index = y + (self.size * x)
                    matrix_A[state_index, new_state_index] -= policy[action_type] * self.gamma
                    matrix_B[state_index] += policy[action_type] * reward

        return np.round(np.linalg.solve(matrix_A, matrix_B).reshape((self.size, self.size)), decimals=1)

    
if __name__ == '__main__':
    grid_size = 5
    A = np.array([0, 1])
    A_prime = np.array([4, 1])
    B = np.array([0, 3])
    B_prime = np.array([2, 3])
    policy = {
        'left': 0.25,
        'right': 0.25,
        'up': 0.25,
        'down': 0.25
    }
    discount = 0.9
    gw = GridWorld(grid_size, A, A_prime, B, B_prime, discount)
    value_map = gw.find_value_function_given_policy_by_matrix(policy)
    print(value_map)
        

