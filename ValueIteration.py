'''
Environment: Windy World
Method: Value iteration (Bellman Optimality Equation as an update rule)
'''

from WindyGridworld import WindyGridworld,get_penalized_windygridworld,get_windygridworld
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

def get_probs_rewards(g):
    probabilities = {}
    rewards = {}
    for (state,action),v in g.transition_probabilities.items():
        for s_prime,p in v.items():
            probabilities[(s_prime,state,action)] = p
            rewards[s_prime] = g.rewards.get(s_prime,0)
    return probabilities,rewards

if __name__ == '__main__':
    '''
    |   |   |   | 1 |
    |   | x |   |-1 |
    | S | 0 |   |   |
    '''
    g = get_windygridworld()
    transition_probabilities,rewards = get_probs_rewards(g)

    # deterministic policy
    policy = {}
    for state in g.actions:
        policy[state] = np.random.choice(g.actions[state])

    print_policy(policy,g)

    #reset all state value
    V = {}
    for state in g.get_all_states():
        V[state] = 0

    # Value iteration
    '''
    Combine policy evaluation, with policy improvement in the same loop
    Now update the optimal Value, not all Value using argmax_a
    '''
    threshold = 1e-3
    gamma = 0.9
    while True:
        max_diff = 0
        for state in g.actions:
            V_old = V[state]
            best_value = float('-inf')
            #argmax_a V
            for action in g.actions[state]:
                Vs_a = 0
                for s_prime in g.get_all_states():
                    tr_p = transition_probabilities.get((s_prime,state,action),0)
                    r = rewards.get(s_prime,0)
                    Vs_a += tr_p * (r + gamma * V.get(s_prime,0))
                if best_value < Vs_a:
                    best_value = Vs_a
                V[state] = best_value

            max_diff = max(max_diff,np.abs(best_value-V_old))
        if max_diff < threshold:
            break

    '''
    We can choose the policy at the end, already having the value function
    '''
    ## choose policy
    for state in g.actions:
        best_value = float('-inf')
        new_policy = None
        #argmax_a
        for action in g.actions[state]:
            Vs_a = 0
            for s_prime in g.get_all_states():
                tr_p = transition_probabilities.get((s_prime,state,action),0)
                r = rewards.get(s_prime,0)
                Vs_a += tr_p * (r + gamma * V.get(s_prime,0))
            if best_value < Vs_a:
                best_value = Vs_a
                new_policy = action
        policy[state] = new_policy

    print("\nValue Iteration:")
    print_policy(policy,g)
    print_value(V,g)
