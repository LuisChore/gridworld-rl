
class Gridworld():
    def __init__(self,n,m,start_position):
        self.I = start_position[0]
        self.J = start_position[1]
        self.n = n
        self.m = m
        self.r = [-1,0,1,0] #U,R,D,L
        self.c = [0,1,0,-1]

    def current_state(self):
        return self.I,self.J

    #dynamic programming purpose
    def next_state(self,state,action):
        I,J = state
        I = I + self.r[action]
        J = J + self.c[action]
        return (I,J)

    def set(self,rewards,walls,terminal_states,actions):
        self.rewards = rewards
        self.walls = walls
        self.terminal_states = terminal_states
        self.actions = actions

    #testing purpose
    def set_state(self,state):
        self.I,self.J = state

    def move(self,action):
        self.I = self.I + self.r[action]
        self.J = self.J + self.c[action]
        return self.rewards.get((self.I,self.J),0)

    def game_over(self):
        return (self.I,self.J) in self.terminal_states

def get_gridworld():
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

    g = Gridworld(n,m,start_position)
    g.set(rewards,walls,terminal_states,actions)
    return g
