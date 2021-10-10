# REINFORCEMENT LEARNING ALGORITHMS

Implementation of reinforcement learning algorithms to solve the gridworld problem

## Gridworld environment
Implementation found in `Gridworld.py` as an NxM matrix that represents a Gridworld environment.

- States: Coordinates in the gridworld
   - Initial/Terminal states: Indicated in the Environment Constructor
   - Walls (forbidden states): Indicated in the Environment Constructor
- Actions: Up/Down/Left/Right
- Rewards: Each state has its own reward

## Dynamic Programming

Prediction Problem
- Iterative Policy Evaluation
  - Input: ![equation](https://latex.codecogs.com/gif.latex?%5Cpi%20%28a%5Cmid%20s%29)
  - Output: ![equation](https://latex.codecogs.com/gif.latex?V_%7B%5Cpi%7D%28s%29)  

Control Problem
- Policy Improvement
- Policy Iteration
- Value Iteration


#### Iteration Policy Evaluation (IPE)

We assume that we know the _dynamic environment_ distribution  ![equation](https://latex.codecogs.com/gif.latex?p%28s%27%2Cr%20%5Cmid%20s%2Ca%29) in the Value Function.

![equation](https://latex.codecogs.com/gif.latex?V_%7B%5Cpi%7D%28s%29%20%3D%20%5Csum_%7Ba%7D%5Cpi%20%28a%5Cmid%20s%29%20%5Csum_%7Bs%27%7D%5Csum_%7Br%7D%20p%28s%27%2Cr%20%5Cmid%20s%2Ca%29%5Cleft%20%5C%7B%20r%20&plus;%20%5Cgamma%20V_%5Cpi%28s%27%29%5Cright%20%5C%7D)


To apply  Iteration Policy Evaluation (IPV) notice that the only unknown variables are the _V's_, to solve it with dynamic programming we only aplly the Bellman Equation (Value Function) over and over again.

1. Initialize:

![equation](https://latex.codecogs.com/gif.latex?V_0%28s%29%20%3D%200%2C%20%5Cforall%20s%20%5Cin%20S)
2. Repeate for k = 1, 2, ... until convergence:

![equation](https://latex.codecogs.com/gif.latex?V_%7Bk&plus;1%7D%28s%29%20%3D%20%5Csum_%7Ba%7D%5Cpi%20%28a%5Cmid%20s%29%20%5Csum_%7Bs%27%7D%5Csum_%7Br%7D%20p%28s%27%2Cr%20%5Cmid%20s%2Ca%29%5Cleft%20%5C%7B%20r%20&plus;%20%5Cgamma%20V_k%28s%27%29%5Cright%20%5C%7D)

Proof about this covergence: Banach fixed point theorem


Implementation in `IPE_deterministic.py`
**Pseudocode**
![pseudocode](https://i.ibb.co/jGCR1SZ/IPE.png)
