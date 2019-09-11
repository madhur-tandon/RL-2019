import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def poisson(lambd, n):
    return (lambd ** n) * math.exp(-lambd) / math.factorial(n)

class CarRental:
    def __init__(self, max_cars, max_move, reward_per_car,
                 cost_to_move_car, request_loc1_lambda,
                 request_loc2_lambda, return_loc1_lambda,
                 return_loc2_lambda, night_keeping_cost, gamma, extended_conditions=True):

        self.max_cars = max_cars
        self.max_move = max_move
        self.reward_per_car = reward_per_car
        self.cost_to_move_car = cost_to_move_car
        self.request_loc1_lambda = request_loc1_lambda
        self.request_loc2_lambda = request_loc2_lambda
        self.return_loc1_lambda = return_loc1_lambda
        self.return_loc2_lambda = return_loc2_lambda
        self.night_keeping_cost = night_keeping_cost
        self.gamma = gamma
        self.all_actions = np.arange(-self.max_move, self.max_move+1)
        self.inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(self.all_actions)}
        self.value_map = np.zeros((self.max_cars+1, self.max_cars+1))
        self.policy_map = np.zeros((self.max_cars+1, self.max_cars+1))
        self.extended_conditions = extended_conditions

    def get_value(self, state, action, value_map):
        value = 0

        num_cars_loc1, num_cars_loc2 = state
        num_cars_loc1 -= action
        num_cars_loc2 += action

        if self.extended_conditions:
            if action>0:
                action -=1
            cost = self.cost_to_move_car * abs(action)
            if num_cars_loc1 > self.max_cars / 2:
                cost += self.night_keeping_cost
            if num_cars_loc2 > self.max_cars / 2:
                cost += self.night_keeping_cost
        else:
            cost = self.cost_to_move_car * abs(action)

        all_possible_requests_loc1 = np.arange(num_cars_loc1+1)
        all_possible_requests_loc2 = np.arange(num_cars_loc2+1)

        if len(all_possible_requests_loc1)>1:
            all_probs_requests_loc1 = [poisson(self.request_loc1_lambda, i) for i in all_possible_requests_loc1[:-1]]
            all_probs_requests_loc1.append(1 - np.sum(all_probs_requests_loc1))
        else:
            all_probs_requests_loc1 = [1]

        if len(all_possible_requests_loc2)>1:
            all_probs_requests_loc2 = [poisson(self.request_loc2_lambda, i) for i in all_possible_requests_loc2[:-1]]
            all_probs_requests_loc2.append(1 - np.sum(all_probs_requests_loc2))
        else:
            all_probs_requests_loc2 = [1]

        for i, each_request_loc1 in enumerate(all_possible_requests_loc1):
            each_reward_loc1 = i * self.reward_per_car
            for j, each_request_loc2 in enumerate(all_possible_requests_loc2):
                each_reward_loc2 = j * self.reward_per_car

                all_possible_returns_loc1 = np.arange(self.max_cars + i - num_cars_loc1 + 1)
                all_possible_returns_loc2 = np.arange(self.max_cars + j - num_cars_loc2 + 1)

                if len(all_possible_returns_loc1)>1:
                    all_probs_returns_loc1 = [poisson(self.return_loc1_lambda, i) for i in all_possible_returns_loc1[:-1]]
                    all_probs_returns_loc1.append(1 - np.sum(all_probs_returns_loc1))
                else:
                    all_probs_returns_loc1 = [1]

                if len(all_possible_returns_loc2)>1:
                    all_probs_returns_loc2 = [poisson(self.return_loc2_lambda, i) for i in all_possible_returns_loc2[:-1]]
                    all_probs_returns_loc2.append(1 - np.sum(all_probs_returns_loc2))
                else:
                    all_probs_returns_loc2 = [1]

                for k, each_return_loc1 in enumerate(all_possible_returns_loc1):
                    for l, each_return_loc2 in enumerate(all_possible_returns_loc2):
                        prob = all_probs_requests_loc1[i] * all_probs_requests_loc2[j] * all_probs_returns_loc1[k] * all_probs_returns_loc2[l]
                        reward = each_reward_loc1 + each_reward_loc2

                        final_num_cars_loc1 = int(num_cars_loc1 - i + k)
                        final_num_cars_loc2 = int(num_cars_loc2 - j + l)

                        value += prob * (reward - cost + (self.gamma * value_map[final_num_cars_loc1][final_num_cars_loc2]))

        return value

    def arg_max(self, arr):
        max_val = np.amax(arr)
        max_val_indices = np.where(arr == max_val)[0]
        return np.random.choice(max_val_indices)

    def plot_policy(self, n):
        plt.figure()
        plt.imshow(self.policy_map, origin='lower', interpolation='none', vmin=-self.max_move, vmax=self.max_move)
        plt.xlabel('#Cars at second location')
        plt.ylabel('#Cars at first location')
        plt.colorbar()
        plt.savefig('{}.png'.format(n))

    def plot_values(self, n):
        X = np.arange(0, self.max_cars + 1)
        Y = X.copy()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, self.value_map)
        plt.savefig('{}.png'.format(n))

    def policy_evaluation(self):
        while True:
            delta = 0
            for i in range(self.max_cars+1):
                for j in range(self.max_cars+1):
                    state = np.array([i, j])
                    v = self.value_map[i, j]
                    self.value_map[i, j] = self.get_value(state, self.policy_map[i, j], self.value_map)
                    delta = max(delta, np.abs(v - self.value_map[i, j]))
            print(delta)
            if delta < 1e-1:
                break
        
    def policy_improvement(self):
        while True:
            self.policy_evaluation()
            policy_stable = True
            for i in range(self.max_cars+1):
                for j in range(self.max_cars+1):
                    all_action_values = {}
                    state = np.array([i, j])
                    old_action = self.policy_map[i, j]
                    for each_action in self.all_actions:
                        if (i - each_action < 0) or (j + each_action < 0) or (i - each_action > self.max_cars) or (j + each_action > self.max_cars):
                            pass
                        else:
                            all_action_values[each_action] = self.get_value(state, each_action, self.value_map)
                    max_value = max(all_action_values.values())
                    max_value_indices = []
                    for each_action, action_value in all_action_values.items():
                        if action_value == max_value:
                            max_value_indices.append(each_action)
                    self.policy_map[i, j] = np.random.choice(max_value_indices)
                    
                    if old_action != self.policy_map[i, j]:
                        policy_stable = False

            if policy_stable:
                break
        

if __name__ == '__main__':
    CR = CarRental(20, 5, 10, 2, 3, 4, 3, 2, 4, 0.9)
    CR.policy_improvement()
    CR.plot_values('hello-values')
    CR.plot_policy('hello')

