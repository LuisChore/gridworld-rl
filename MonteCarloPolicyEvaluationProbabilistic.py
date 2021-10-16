'''
Environment: GridWorld
Method: Monte Carlo Policy Evaluation
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


# probabilistic -> windy policy
def random_action(g,state,action):
    p = np.random.random()
    if p < 0.5:
        return action
    else:
        return np.random.choice(g.actions[state])

def play_game(g,policy):
    gamma = 0.9
    #returns a list of states and returns
    '''
    Let's choose randomly a start state and action
    in order to visit all states
    '''
    possible_states = g.actions.keys()
    index = np.random.choice(len(possible_states))
    g.set_state(list(possible_states)[index])

    s = g.current_state()
    states_rewards = [(s,0)] # 0 reward for being in s_0
    # we can limit the number of steps
    limit = 100
    cycle_cost = -10000
    while not g.game_over():
        limit -= 1
        if limit == 0:
            print('policy failed')
            states_rewards.append((s,cycle_cost))
            break
        a = policy[s]
        a = random_action(g,s,a) # WINDY
        r = g.move(a)
        s = g.current_state()
        states_rewards.append((s,r))

    G = 0 # final state return
    states_returns = []
    # backwards to compute recursevely
    states_rewards.reverse()
    first = True # don't save final state value, it's always 0
    for s,r in states_rewards:
        if first:
            first = False
        else:
            states_returns.append((s,G))
        G = r + gamma * G
    states_returns.reverse()
    return states_returns

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
        returns[state] = []

    ## FIRST SEEN Monte Carlo EVALUATION
    iterations = 100
    for it in range(iterations):
        states_seen = set()
        states_returns = play_game(g,policy)
        for s,G in states_returns:
            if s not in states_seen:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                states_seen.add(s)
    print_value(V,g)
