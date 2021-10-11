import numpy as np

class WindyGridworld():
    def __init__(self,n,m,start_position):
        self.I = start_position[0]
        self.J = start_position[1]
        self.n = n
        self.m = m
        self.r = [-1,0,1,0] #U,R,D,L
        self.c = [0,1,0,-1]

    def current_state(self):
        return self.I,self.J

    def get_all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def set(self,rewards,walls,terminal_states,actions,transition_probabilities):
        self.rewards = rewards
        self.walls = walls
        self.terminal_states = terminal_states
        self.actions = actions
        self.transition_probabilities = transition_probabilities

    #function to move in a windy gridworld
    def move(self,action):
        state = (self.I,self.J)
        probs = self.transition_probabilities.get((state,action))
        s_primes = list(probs.keys())
        prob_values = list(probs.values())

        probabilistic_action = np.random.choice(s_primes,p = prob_values)
        self.I = self.I + self.r[probabilistic_action]
        self.J = self.J + self.c[probabilistic_action]
        return self.rewards.get((self.I,self.J),0)

    def game_over(self):
        return (self.I,self.J) in self.terminal_states


def get_windygridworld():
    n = 3
    m = 4
    start_position = (2,0)
    terminal_states = ((0,3),(1,3))
    walls = ((1,1))
    rewards = {
        (0,3):1,
        (1,3):-1,
    }
    actions = {
        (0,0):(1,2),
        (0,1):(1,3),
        (0,2):(1,2,3),
        (1,0):(0,2),
        (1,2):(0,1,2),
        (2,0):(0,1),
        (2,1):(1,3),
        (2,2):(0,1,3),
        (2,3):(0,3),
    }

    # p(s'|s,a)
    # Key: (s,a) -> Value: {s': p(s',s,a)}
    transition_probabilities = {
        ((0,0),1): {(0,1):1.0},
        ((0,0),2): {(1,0):1.0},
        ((0,1),1): {(0,2):1.0},
        ((0,1),3): {(0,0):1.0},
        ((0,2),1): {(0,3):1.0},
        ((0,2),2): {(1,2):1.0},
        ((0,2),3): {(0,1):1.0},
        ((1,0),0): {(0,0):1.0},
        ((1,0),2): {(2,0):1.0},
        ((1,2),0): {(0,2):0.5,(1,3):0.5},
        ((1,2),1): {(1,3):1.0},
        ((1,2),2): {(2,2):1.0},
        ((2,0),0): {(1,0):1.0},
        ((2,0),1): {(2,1):1.0},
        ((2,1),1): {(2,2):1.0},
        ((2,1),3): {(2,0):1.0},
        ((2,2),0): {(1,2):1.0},
        ((2,2),1): {(2,3):1.0},
        ((2,2),3): {(2,1):1.0},
        ((2,3),0): {(1,3):1.0},
        ((2,3),3): {(2,2):1.0},
    }

    g = WindyGridworld(n,m,start_position)
    g.set(rewards,walls,terminal_states,actions,transition_probabilities)
    return g


def get_penalized_windygridworld(step_cost = 0):
    n = 3
    m = 4
    start_position = (2,0)
    terminal_states = ((0,3),(1,3))
    walls = ((1,1))

    rewards = {}
    for i in range(n):
        for j in range(m):
            rewards[(i,j)] = step_cost

    rewards[(0,3)] = 1
    rewards[(1,3)] = -1

    actions = {
        (0,0):(1,2),
        (0,1):(1,3),
        (0,2):(1,2,3),
        (1,0):(0,2),
        (1,2):(0,1,2),
        (2,0):(0,1),
        (2,1):(1,3),
        (2,2):(0,1,3),
        (2,3):(0,3),
    }

    # p(s'|s,a)
    # Key: (s,a) -> Value: {s': p(s',s,a)}
    transition_probabilities = {
        ((0,0),1): {(0,1):1.0},
        ((0,0),2): {(1,0):1.0},
        ((0,1),1): {(0,2):1.0},
        ((0,1),3): {(0,0):1.0},
        ((0,2),1): {(0,3):1.0},
        ((0,2),2): {(1,2):1.0},
        ((0,2),3): {(0,1):1.0},
        ((1,0),0): {(0,0):1.0},
        ((1,0),2): {(2,0):1.0},
        ((1,2),0): {(0,2):0.5,(1,3):0.5},
        ((1,2),1): {(1,3):1.0},
        ((1,2),2): {(2,2):1.0},
        ((2,0),0): {(1,0):1.0},
        ((2,0),1): {(2,1):1.0},
        ((2,1),1): {(2,2):1.0},
        ((2,1),3): {(2,0):1.0},
        ((2,2),0): {(1,2):1.0},
        ((2,2),1): {(2,3):1.0},
        ((2,2),3): {(2,1):1.0},
        ((2,3),0): {(1,3):1.0},
        ((2,3),3): {(2,2):1.0},
    }

    g = WindyGridworld(n,m,start_position)
    g.set(rewards,walls,terminal_states,actions,transition_probabilities)
    return g
