import numpy as np
import matplotlib.pyplot as plt

true_values = np.arange(0, 7) / 6
true_values[6] = 0

def TD(value_map, alpha):
    current_state = 3

    while True:
        old_state = current_state
        coin_toss = np.random.binomial(1, 0.5)
        if coin_toss == 0:
            current_state -= 1
        else:
            current_state += 1
        
        if current_state == 6:
            reward = 1
        else:
            reward = 0

        value_map[old_state] += alpha * (reward + value_map[current_state] - value_map[old_state])

        if current_state in [0, 6]:
            break
    
    return value_map

def MC(value_map, alpha):
    current_state = 3
    history = [current_state]

    while True:
        coin_toss = np.random.binomial(1, 0.5)
        if coin_toss == 0:
            current_state -= 1
        else:
            current_state += 1
        
        history.append(current_state)

        if current_state == 6:
            cumulative_returns = 1
            break
        elif current_state == 0:
            cumulative_returns = 0
            break

    for each_state in history[:-1]:
        value_map[each_state] += alpha * (cumulative_returns - value_map[each_state])

        

if __name__ == '__main__':
    ### For 1st plot
    all_episodes = [0, 1, 10, 100]
    value_map = np.zeros(7)
    value_map[1:6] = np.full(5, 0.5)
    plt.figure()
    for each_episode in range(np.amax(all_episodes)+1):
        if each_episode in all_episodes:
            plt.plot(value_map[1:6], label='{0} episodes'.format(each_episode))
        TD(value_map, 0.1)
    plt.xticks(np.arange(5), ('A', 'B', 'C', 'D', 'E'))
    plt.plot(true_values[1:6], label='True Values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value')
    plt.legend()
    plt.show()

    ### For 2nd plot
    alphas = {
        'TD': [0.05, 0.1, 0.15],
        'MC': [0.01, 0.02, 0.03, 0.04]
    }
    num_episodes = 101
    num_runs = 100

    all_alphas = np.concatenate(list(alphas.values()), axis=None)

    for each_alpha in all_alphas:
        total_error_for_all_runs = np.zeros(num_episodes)
        for each_run in range(num_runs):
            errors = []
            value_map = np.zeros(7)
            value_map[1:6] = np.full(5, 0.5)
            for each_episode in range(num_episodes):
                rmse_error = np.sqrt(np.sum(np.power((value_map - true_values), 2)) / 5)
                errors.append(rmse_error)
                if each_alpha in alphas['TD']:
                    TD(value_map, each_alpha)
                elif each_alpha in alphas['MC']:
                    MC(value_map, each_alpha)
            total_error_for_all_runs += np.array(errors)
        total_error_for_all_runs /= num_runs
        if each_alpha in alphas['TD']:
            plt.plot(total_error_for_all_runs, linestyle='solid', label='TD' + ', alpha = %.02f' % (each_alpha))
        else:
            plt.plot(total_error_for_all_runs, linestyle='dashdot', label='MC' + ', alpha = %.02f' % (each_alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend(loc='best')
    plt.show()




    