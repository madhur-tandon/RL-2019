import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# 0 means hit aka pick cards
# 1 means stick aka stop picking cards
policy_player = np.zeros(22, dtype=np.int)
policy_player[20:] = 1

policy_dealer = np.zeros(22, dtype=np.int)
policy_dealer[17:] = 1

# picking an ace gives value as 11
# picking face cards give value as 10
all_card_values_with_usable_ace = np.array([11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

def comp(a, b):
    return int(a > b) - int(a < b)

def initialize(initial_state=None):
    if initial_state is not None:
        player_card_sum, player_usable_ace, dealer_card_face_up = initial_state
        dealer_card_2 = np.random.choice(all_card_values_with_usable_ace)
        dealer_card_sum = dealer_card_face_up + dealer_card_2
        dealer_usable_ace = (11 in [dealer_card_face_up, dealer_card_2])
        if dealer_card_sum > 21:
            dealer_card_sum -= 10
        state = initial_state
        player_card_history = []
    else:
        player_card_sum, dealer_card_sum = 0, 0
        player_usable_ace, dealer_usable_ace = False, False
        player_card_history = []

        while player_card_sum < 12:
            card = np.random.choice(all_card_values_with_usable_ace)
            player_card_sum += card

            if player_card_sum > 21:
                # the previous sum would have been 11 before adding a new card
                # (since if previous sum is < 11 (suppose 10), then new sum is at max 21
                #  as 11 is the card with the highest value, which doesn't violate the condition)
                # the new card has to be 11 for sum to exceed 21, which makes the new sum as 22
                # this makes us go bust, and thus, the last ace's value should be taken as 1
                player_card_sum -= 10
            else:
                player_usable_ace = player_usable_ace | (card == 11)
    
        dealer_initial_cards = np.random.choice(all_card_values_with_usable_ace, 2)
        dealer_card_sum += np.sum(dealer_initial_cards)
        dealer_usable_ace = dealer_usable_ace | (11 in dealer_initial_cards)

        if dealer_card_sum > 21:
            dealer_card_sum -= 10

        dealer_card_face_up = np.random.choice(dealer_initial_cards)
        state = (player_card_sum, player_usable_ace, dealer_card_face_up)

    return player_card_sum, dealer_card_sum, player_usable_ace, dealer_usable_ace, dealer_card_face_up, state, player_card_history

def simulate_player_turn(state, player_card_history, policy=None, initial_action=None, **kwargs):
    player_card_sum, player_usable_ace, dealer_card_face_up = state

    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            if policy is None:
                action = policy_player[player_card_sum]
            else:
                action = policy(player_card_sum, player_usable_ace, dealer_card_face_up, **kwargs)
        
        new_state = (player_card_sum, player_usable_ace, dealer_card_face_up)
        player_card_history.append([new_state, action])

        # stick
        if action == 1:
            return 'stick', player_card_sum, new_state, player_card_history
        
        # hit
        else:
            card = np.random.choice(all_card_values_with_usable_ace)
            num_aces = 1 if player_usable_ace else 0
            if card == 11:
                num_aces += 1
            player_card_sum += card

            while player_card_sum > 21 and num_aces > 0:
                player_card_sum -= 10
                num_aces -= 1

            if player_card_sum > 21:
                reward = -1
                return state, reward, player_card_history
            
            if num_aces == 1:
                player_usable_ace = True
            else:
                player_usable_ace = False

def simulate_dealer_turn(state, dealer_card_sum, dealer_usable_ace, player_card_history):
    while True:
        action = policy_dealer[dealer_card_sum]

        # stick
        if action == 1:
            return 'stick', dealer_card_sum, state, player_card_history
        
        # hit
        else:
            card = np.random.choice(all_card_values_with_usable_ace)
            num_aces = 1 if dealer_usable_ace else 0
            if card == 11:
                num_aces += 1
            dealer_card_sum += card

            while dealer_card_sum > 21 and num_aces > 0:
                dealer_card_sum -= 10
                num_aces -= 1

            if dealer_card_sum > 21:
                reward = 1
                return state, reward, player_card_history
            
            if num_aces == 1:
                dealer_usable_ace = True
            else:
                dealer_usable_ace = False
    
def simulate_play(policy=None, initial_state=None, initial_action=None, **kwargs):
    player_card_sum, dealer_card_sum, player_usable_ace, \
    dealer_usable_ace, dealer_card_face_up, \
    state, player_card_history = initialize(initial_state)

    player_turn_result = simulate_player_turn(state, player_card_history, policy, initial_action, **kwargs)
    if len(player_turn_result) == 4:
        _, player_card_sum, state, player_card_history = player_turn_result
    else:
        return player_turn_result

    dealer_turn_result = simulate_dealer_turn(state, dealer_card_sum, dealer_usable_ace, player_card_history)
    if len(dealer_turn_result) == 4:
        _, dealer_card_sum, state, player_card_history = dealer_turn_result
    else:
        return dealer_turn_result

    reward = comp(player_card_sum, dealer_card_sum)
    return state, reward, player_card_history

def argmax_policy(player_card_sum, player_usable_ace, dealer_card_face_up, **kwargs):
    state_actions = kwargs['state_actions']
    state_actions_count = kwargs['state_actions_count']
    if dealer_card_face_up == 11:
        dealer_card_face_up = 1
    state_action_values = state_actions[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1, :] / \
        state_actions_count[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1, :]
    max_state_action_value = np.amax(state_action_values)
    all_actions_with_max_value = [i for i, a in enumerate(state_action_values) if a == max_state_action_value]
    return np.random.choice(all_actions_with_max_value)

def MC_on_policy(num_episodes):
    states = np.zeros((2, 10, 10))
    states_ace_count = np.ones((2, 10, 10))

    for i in range(num_episodes):
        _, reward, player_history = simulate_play()
        for (state, _) in player_history:
            player_card_sum, player_usable_ace, dealer_card_face_up = state
            if dealer_card_face_up == 11:
                dealer_card_face_up = 1
            states_ace_count[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1] += 1
            states[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1] += reward
    
    return states / states_ace_count

def MC_ES(num_episodes):
    state_actions = np.zeros((2, 10, 10, 2))
    state_actions_count = np.full((2, 10, 10, 2), 1)
    
    for i in range(num_episodes):
        random_usable_ace = bool(np.random.choice([0, 1]))
        random_player_card_sum = np.random.choice(np.arange(12, 22))
        random_dealer_card_face_up = np.random.choice(all_card_values_with_usable_ace)
        random_initial_state = (random_player_card_sum, random_usable_ace, random_dealer_card_face_up)
        random_initial_action = np.random.choice([0, 1])

        if i==0:
            policy = None
        else:
            policy = argmax_policy
        
        _, reward, player_history = simulate_play(policy, random_initial_state, random_initial_action, **{'state_actions': state_actions, 'state_actions_count': state_actions_count})
        for (state, action) in player_history:
            player_card_sum, player_usable_ace, dealer_card_face_up = state
            if dealer_card_face_up == 11:
                dealer_card_face_up = 1
            state_actions[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1, action] += reward
            state_actions_count[int(player_usable_ace), player_card_sum-12, dealer_card_face_up-1, action] += 1
        
    return state_actions / state_actions_count

def MC_off_policy(num_episodes):
    def behaviour_policy_for_player(player_card_sum, player_usable_ace, dealer_card_face_up):
        coin_toss = np.random.binomial(1, 0.5)
        if coin_toss == 1:
            return 1
        else:
            return 0

    initial_state = (13, True, 2)

    all_ratios = np.zeros(num_episodes)
    all_returns = np.zeros(num_episodes)

    for i in range(num_episodes):
        _, reward, player_history = simulate_play(behaviour_policy_for_player, initial_state)
        num, den = 1, 1
        for (state, action) in player_history:
            player_card_sum, player_usable_ace, dealer_card_face_up = state
            target_action = policy_player[player_card_sum]
            if action == target_action:
                den *= 0.5
            else:
                num = 0
                break
        all_ratios[i] = num / den
        all_returns[i] = reward

    weighted_returns = np.cumsum(all_ratios * all_returns)
    all_ratios = np.cumsum(all_ratios)
    T_S = np.arange(1, num_episodes+1)

    ordinary_sampling = weighted_returns / T_S
    weighted_sampling = np.where(all_ratios!=0, weighted_returns / all_ratios, 0)

    return ordinary_sampling, weighted_sampling

def fig_1():
    states_1 = MC_on_policy(10000)
    states_2 = MC_on_policy(500000)

    titles = ['Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes',
              'No Usable Ace, 500000 Episodes']

    states_order = [states_1[1], states_2[1], states_1[0], states_2[0]]

    fig, ax = plt.subplots(2, 2, figsize=(40, 30), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    card_labels = ['A']
    card_labels.extend(range(2, 11, 1))
    for i in range(2):
        for j in range(2):
            index = i*2 + j
            xs, ys = np.meshgrid(range(10), range(10))
            ax[i, j].set_zlim(-1, 1)
            ax[i, j].set_xticks(ticks=range(0, 10, 1))
            ax[i, j].set_xticklabels(card_labels)
            ax[i, j].set_yticks(ticks=range(0, 10, 1))
            ax[i, j].set_yticklabels(range(12, 22, 1))
            ax[i, j].set_xlabel('Dealer Showing')
            ax[i, j].set_ylabel('Player Sum')
            ax[i, j].set_title(titles[index])
            ax[i, j].plot_surface(xs, ys, states_order[index], rstride=1, cstride=1, cmap='hot')
    
    plt.show()

def fig_2():
    state_action_values = MC_ES(500000)

    state_action_values_non_usable_ace = state_action_values[0]
    state_action_values_usable_ace = state_action_values[1]

    state_values_non_usable_ace = np.amax(state_action_values_non_usable_ace, axis=2)
    state_values_usable_ace = np.amax(state_action_values_usable_ace, axis=2)
    
    action_non_usable_ace = np.argmax(state_action_values_non_usable_ace, axis=2)
    action_usable_ace = np.argmax(state_action_values_usable_ace, axis=2)

    plots = [action_usable_ace, state_values_usable_ace, action_non_usable_ace, state_values_non_usable_ace]

    card_labels = ['A']
    card_labels.extend(range(2, 11, 1))
    xs, ys = np.meshgrid(range(10), range(10))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.set_xticks(ticks=range(0, 10, 1))
    ax.set_xticklabels(card_labels)
    ax.set_yticks(ticks=range(0, 10, 1))
    ax.set_yticklabels(range(21, 11, -1))
    ax.set_title('Usable Ace PI*')
    ax.imshow(np.flipud(plots[0]), cmap='YlGnBu')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_zlim(-1, 1)
    ax.set_xticks(ticks=range(0, 10, 1))
    ax.set_xticklabels(card_labels)
    ax.set_yticks(ticks=range(0, 10, 1))
    ax.set_yticklabels(range(12, 22, 1))
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title('Usable Ace V*')
    ax.plot_surface(xs, ys, plots[1], rstride=1, cstride=1, cmap='hot')

    ax = fig.add_subplot(2, 2, 3)
    ax.set_xticks(ticks=range(0, 10, 1))
    ax.set_xticklabels(card_labels)
    ax.set_yticks(ticks=range(0, 10, 1))
    ax.set_yticklabels(range(21, 11, -1))
    ax.set_title('No Usable Ace PI*')
    ax.imshow(np.flipud(plots[2]), cmap='YlGnBu')

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_zlim(-1, 1)
    ax.set_xticks(ticks=range(0, 10, 1))
    ax.set_xticklabels(card_labels)
    ax.set_yticks(ticks=range(0, 10, 1))
    ax.set_yticklabels(range(12, 22, 1))
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title('No Usable Ace V*')
    ax.plot_surface(xs, ys, plots[3], rstride=1, cstride=1, cmap='hot')

    plt.show()

def fig_3():
    np.random.seed(42)
    true_val = -0.27726
    num_episodes = 10000
    num_runs = 100

    ordinary_sampling_errors = np.zeros(num_episodes)
    weighted_sampling_errors = np.zeros(num_episodes)

    for each_run in range(num_runs):
        ordinary_sampling, weighted_sampling = MC_off_policy(num_episodes)
        ordinary_error = np.power(ordinary_sampling - true_val, 2)
        weighted_error = np.power(weighted_sampling - true_val, 2)
        ordinary_sampling_errors += ordinary_error
        weighted_sampling_errors += weighted_error

    ordinary_sampling_errors /= num_runs
    weighted_sampling_errors /= num_runs

    plt.figure()
    plt.plot(ordinary_sampling_errors, label='Ordinary Importance Sampling')
    plt.plot(weighted_sampling_errors, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend(loc='best')
    plt.ylim([-0.5, 8])
    plt.show()

if __name__ == '__main__':
    fig_1()
    fig_2()
    fig_3()