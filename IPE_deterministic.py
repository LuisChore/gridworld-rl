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

###### ITERATIVE POLICY EVALUATION #####
# Apply Bellman Equation until convergence

def iterative_policy_evaluation(g,policy,threshold):
    V = {}
    threshold_convergence =  threshold
    gamma = 0.9 # discount factor

    #initialize Value Function
    for state in policy:
        V[state] = 0

    while True:
        max_diff = 0
        for state in policy:
            oldVs = V[state]
            Vs = 0
            for action in g.actions[state]:
                # Probability pi, Policy deterministic
                PI = 1 if policy.get(state) == action else 0
                # Environment deterministic
                s_prime = g.next_state(state,action)
                transition_probability = 1
                Vs_prime = V.get(s_prime,0)
                # Reward deterministic
                r = g.rewards.get(s_prime,0)
                #Bellman Equation
                Vs += PI * transition_probability * ( r + gamma * Vs_prime)

            max_diff = max(max_diff, np.abs(Vs - oldVs))
            V[state] = Vs
        if max_diff < threshold:
            break
    return V




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
    V = iterative_policy_evaluation(g,policy,1e-3)
    print_value(V,g)
