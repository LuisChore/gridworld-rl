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


def get_transitionprobs_reward(g):
    probabilities = {}
    rewards = {}
    for state in g.actions:
        for action in g.actions[state]:
            s_prime = g.next_state(state,action)
            # envvironment deterministic
            probabilities[(s_prime,state,action)] = 1
            rewards[s_prime] = g.rewards.get(s_prime,0)

    return probabilities,rewards

#it recieves an already initialized V for a faster convergence
def policy_evaluation(g,policy,threshold,probabilities,rewards,V):
    gamma = 0.9

    while True:
        max_diff = 0
        for state in g.actions:
            oldVs = V[state]
            Vs = 0
            for action in g.actions[state]:
                #pi, policy deterministic
                pi = 1 if policy.get(state) == action else 0
                # environment deterministic
                s_prime = g.next_state(state,action)
                tr_pr = probabilities.get((s_prime,state,action),0)
                Vs_prime = V.get(s_prime,0)
                # reward deterministic
                r = rewards.get(s_prime,0)
                Vs += pi * tr_pr * (r + gamma * Vs_prime)
            V[state] = Vs
            max_diff = max(max_diff,np.abs(Vs - oldVs))
        if max_diff < threshold:
            break
    return V



def policy_iteration(g,transition_probabilities,rewards,policy):
    #Value Function
    V = {}
    for state in policy:
        V[state] = 0

    gamma = 0.9

    while True:
        V = policy_evaluation(g,policy,1e-3,transition_probabilities,rewards,V)
        #POLICY IMPROVEMENT
        policy_stable = True
        for state in g.actions:
            oldVs = V[state]
            old_policy = policy[state]
            best_value = float('-inf')
            best_policy = None
            #argmax_a V
            for action in g.actions[state]:
                # don't use PI(s|a)
                Vs_a = 0
                # environment deterministic
                s_prime = g.next_state(state,action)
                tr_pr = transition_probabilities.get((s_prime,state,action),0)
                Vs_prime = V.get(s_prime,0)
                # reward deterministic
                R = rewards.get(s_prime,0)
                Vs_a =  tr_pr * (R + gamma * Vs_prime)
                if best_value < Vs_a:
                    best_policy = action
                    best_value = Vs_a

            policy[state] = best_policy
            if old_policy != best_policy:
                policy_stable = False
        if policy_stable:
            break
    print_value(V,g)

if __name__ == '__main__':
    '''
    |   |   |   | 1 |
    |   | x |   |-1 |
    | S |   |   |   |
    '''
    g = get_gridworld()
    transition_probabilities, rewards = get_transitionprobs_reward(g)

    #deterministic policy (RANDOM)
    policy = {}
    for state in g.actions:
        policy[state] = np.random.choice(g.actions[state])

    print_policy(policy,g)
    print("POLICY ITERATION")
    policy_iteration(g,transition_probabilities,rewards,policy)
    print_policy(policy,g)
