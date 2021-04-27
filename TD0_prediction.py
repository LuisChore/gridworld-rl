'''
Policy Evaluation using temporal difference
Epsilon Greedy to visit all states
'''

from Gridworld import Gridworld,get_gridworld
import numpy as np

###### VISUALIZATION FUNCTIONS #####
def get_action_value(pi):
    if pi == 0:
        return 'U'
    elif pi == 1:
        return 'R'
    elif pi == 2:
        return 'D'
    elif pi == 3:
        return 'L'
    return '?'

def print_policy(pi,g):
    for i in range(g.n):
        for j in range(g.m):
            pi_s = get_action_value(pi.get((i,j),-1))
            print(" %c " %pi_s ,end = "|")
        print()

def print_value(V,g):
    for i in range(g.n):
        for j in range(g.m):
            v_s = V.get((i,j),0)
            if v_s >= 0:
                print(" %.2f " % v_s,end = '|')
            else:
                print("%.2f " % v_s,end = '|')
        print()

# eps-greedy
def random_action(g,state,action,eps = 0.1):
    if np.random.random() < 1 - eps:
        return action
    else:
        return np.random.choice(g.actions[state])

#deterministic game
def play_game(g,policy):
    s = (2,0)
    g.set_state(s)
    states_rewards = [(s,0)] # 0 reward for being in s_0
    while not g.game_over():
        a = policy[s]
        a = random_action(g,s,a)
        r = g.move(a)
        s = g.current_state()
        states_rewards.append((s,r))
    # it only returns: states, rewards, because everything else would be
    #computed online
    return states_rewards

if __name__ == '__main__':
    '''
    |   |   |   | 1 |
    |   | x |   |-1 |
    | S |   |   |   |
    '''
    g = get_gridworld()

    #deterministic policy
    policy = {
        (0,0):1,
        (0,1):1,
        (0,2):1,
        (1,0):0,
        (1,2):0,
        (2,0):0,
        (2,1):1,
        (2,2):0,
        (2,3):3,
    }
    print_policy(policy,g)

    #initialize Value Functions
    V = {}
    #initialize returns, to compute the mean
    returns = {}
    for state in g.actions:
        V[state] = 0

    # TD(0)
    alpha = 0.1
    gamma = 0.9
    iterations = 1000
    for it in range(iterations):
        states_rewards = play_game(g,policy)
        for i in range(len(states_rewards) - 1):
            s,_ = states_rewards[i]
            s2,r = states_rewards[i + 1]
            td = r + gamma * V.get(s2,0) - V[s]
            #update
            V[s] = V[s] + alpha * td
    print_value(V,g)
