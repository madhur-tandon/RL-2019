import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

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

    def mapping(self, i):
        if i==0:
            return 'left'
        elif i==1:
            return 'right'
        elif i==2:
            return 'up'
        elif i==3:
            return 'down'

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

    def find_optimal_value_function_by_matrix(self):
        num_states = self.size * self.size
        matrix_A = np.zeros((num_states * len(self.all_actions), num_states))
        matrix_B = np.zeros((num_states * len(self.all_actions)))

        for i in range(self.size):
            for j in range(self.size):
                state = np.array([i, j])
                for k, (action_type, each_action) in enumerate(self.all_actions.items()):
                    state_index = len(self.all_actions) * (j + (self.size * i)) + k
                    next_state, reward = self.take_step(state, each_action)
                    x, y = next_state
                    new_state_index = y + (self.size * x)
                    matrix_A[state_index, new_state_index] += self.gamma
                    matrix_B[state_index] -= reward

        for p in range(num_states * len(self.all_actions)):
            q = int(p / 4)
            matrix_A[p, q] -= 1

        c = np.ones(num_states)
        solution = linprog(c, A_ub=matrix_A, b_ub=matrix_B)

        x = (np.sum(matrix_A * np.round(solution['x'], decimals=1), axis=1) + matrix_B).reshape(-1, len(self.all_actions))
        max_x = np.amax(x, axis=1)

        optimal_policy_map = np.empty(num_states, object)
        for k in range(num_states):
            y = np.ravel(np.argwhere(x[k]==max_x[k]))
            for each_action in y:
                if optimal_policy_map[k] is None:
                    optimal_policy_map[k] = [self.mapping(each_action)]
                else:
                    optimal_policy_map[k].append(self.mapping(each_action))
        
        optimal_policy_map = optimal_policy_map.reshape((self.size, self.size))
        optimal_value_map = np.round(np.reshape(solution['x'], (self.size, self.size)), decimals=1)

        return optimal_value_map, optimal_policy_map
    
if __name__ == '__main__':
    grid_size = 5
    A = np.array([0, 1])
    A_prime = np.array([4, 1])
    B = np.array([0, 3])
    B_prime = np.array([2, 3])
    discount = 0.9
    gw = GridWorld(grid_size, A, A_prime, B, B_prime, discount)
    optimal_value_map, optimal_policy_map = gw.find_optimal_value_function_by_matrix()
    print(optimal_value_map)
    for i in range(grid_size):
        for j in range(grid_size):
            print(optimal_policy_map[i][j], end='\t')
        print()

