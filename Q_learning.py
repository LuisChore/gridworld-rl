'''
Environment: Gridworld
EPS-greedy
Temporal difference to update q-value
The use of Q-value to have knowledge about which action is the best
'''
from Gridworld import Gridworld,get_gridworld2,get_gridworld
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
    print("\nQ-Learning:")
    gamma = 0.9
    alpha = 0.1
    Q = {}
    #initialize Q values
    for state in g.get_all_states():
        Q[state] = {}
        if state in g.actions:
            for action in g.actions[state]:
                Q[state][action] = 0

    deltas = [] #keep track biggest changes
    iterations = 2000
    for it in range(iterations):
        max_diff = 0 # testing purpose
        s = (2,0)
        g.set_state(s)
        a = max_dict(Q[s])[0]
        while not g.game_over():
            #it could change even if we update the previous state  with the best action
            a = random_action(g,s,a)

            old_Qsa = Q[s][a]
            r = g.move(a)
            s_prime = g.current_state()
            #temporal difference
            a_prime,Qsa_prime = max_dict(Q[s_prime])

            #Special case, s_prime is a terminal state
            #Q[s_prime][a_prime] is not defined 
            if a_prime == None:
                Qsa_prime = 0

            TD = r + gamma * Qsa_prime - Q[s][a]
            #update Q
            Q[s][a] = Q[s][a] + alpha * TD
            max_diff = max(max_diff,np.abs(old_Qsa-Q[s][a]))
            s,a = s_prime,a_prime

        deltas.append(max_diff)

    #Policy: argmax[a] Q(s,a)
    for state in g.actions:
        policy[state] = max_dict(Q[state])[0]


    plt.plot(deltas)
    plt.show()

    #Compute V(s) using Q(s,a)
    V = {}
    for state in g.actions:
        V[state] = max_dict(Q[state])[1]

    print_policy(policy,g)
    print_value(V,g)
