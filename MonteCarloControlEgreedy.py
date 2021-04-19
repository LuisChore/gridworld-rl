'''
Environment: Gridworld
EPS-greedy
Value Iteration:
    -One iteration for Evaluation
    -Update Q(s,a)
    -Improve Policy argmax[a] Q(s,a)

The use of Q-value to have knowledge about which action is the best
'''
from Gridworld import Gridworld,get_gridworld2
import matplotlib.pyplot as plt
import numpy as np

###### UTIL FUNCTIONS #############
def max_dict(D):
    #return the key and value with the max value
    max_value = float('-inf')
    max_key = None
    for k,v in D.items():
        if v > max_value:
            max_key = k
            max_value = v
    return max_key,max_value

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


# probabilistic -> e-greedy
def random_action(g,state,action,eps = 0.1):
    p = np.random.random()
    if p < (1 - eps):
        return action
    else:
        return np.random.choice(g.actions[state])

def play_game(g,policy):
    gamma = 0.9
    #returns a list of states, actions and returns
    '''eps-greedy
    Initial State (2,0)
    '''

    g.set_state((2,0))
    s = g.current_state()
    a = random_action(g,s,policy[s])
    '''
    Each tripe is s(t),a(t),r(t)
    r(t) results from takin a(t-1) in s(t-1) and landing in s(t)
    '''
    states_actions_rewards = [(s,a,0)] # 0 reward for being in s_0
    while True:
        r = g.move(a)
        s = g.current_state()
        if g.game_over():
            states_actions_rewards.append((s,None,r))
            break
        a = random_action(g,s,policy[s]) # next action
        states_actions_rewards.append((s,a,r))

    G = 0 # final state return
    states_actions_returns = []
    # backwards to compute recursevely
    states_actions_rewards.reverse()
    first = True # don't save final state value, it's always 0
    for s,a,r in states_actions_rewards:
        if first:
            first = False
        else:
            states_actions_returns.append((s,a,G))
        G = r + gamma * G
    states_actions_returns.reverse()
    return states_actions_returns


if __name__ == '__main__':
    '''
    |   |   |   | 1 |
    |   | x |   |-1 |
    | S |   |   |   |
    '''
    g = get_gridworld2()
    # deterministic policy, choosen randomly
    policy = {}
    for state in g.actions:
        policy[state] = np.random.choice(g.actions[state])

    print_policy(policy,g)
    print("\nMonte Carlo:")
    Q = {}
    returns = {}
    #initialize Q values
    #initialize returns, to compute the mean
    for state in g.actions:
        Q[state] = {}
        for action in g.actions[state]:
            returns[(state,action)] = []
            Q[state][action] = 0


    deltas = [] #keep track biggest changes
    iterations = 2000
    for it in range(iterations):
        max_diff = 0 # testing purposes
        # First Visit Monte Carlo Evaluation
        states_actions_returns = play_game(g,policy)
        states_seen = set()
        for s,a,G in states_actions_returns:
            old_Qsa = Q[s][a]
            sa = s,a
            if sa not in states_seen:
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                states_seen.add(sa)
                max_diff = max(max_diff,np.abs(old_Qsa-Q[s][a]))
        #Policy Improvement argmax[a] Q(s,a)
        for state in g.actions:
            policy[state] = max_dict(Q[state])[0]

        deltas.append(max_diff)

    plt.plot(deltas)
    plt.show()

    #Compute V(s) using Q(s,a)
    V = {}
    for state in g.actions:
        V[state] = max_dict(Q[state])[1]

    print_policy(policy,g)
    print_value(V,g)
