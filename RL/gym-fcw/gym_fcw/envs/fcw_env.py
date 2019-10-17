import numpy as np
import sys
from gym.envs.toy_text import discrete


# N - World height
# T - World width
WORLD_HEIGHT = 4
WORLD_WIDTH = 12

UP = 1
ZERO = 0
DOWN = 2
START = [0, 0]


class FcwEnv(discrete.DiscreteEnv):
    """
    This is a simple implementation of the Financial Cliff Walking
    reinforcement learning task.
    Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf
    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [0, 0] as the start at bottom-left
        [0, 11] as the goal at bottom-right
        [0, 1..10] as the cliff at bottom-center
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start. An episode terminates when the agent reaches the goal.
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.shape = (WORLD_HEIGHT, WORLD_WIDTH)
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)
        self.s = 36
        nS = np.prod(self.shape)
        nA = 3

        # Cliff Location (for plotting)
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [1, 1])
            P[s][ZERO] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [-1, 1])

        # Calculate initial state distribution
        # We always start in state (0, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(FcwEnv, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == terminal_state
        return [(1.0, new_state, -1, is_done)]

    def step(self, state, action):
        # Step function that describes how the next state is obtained from the current state and the action 
# taken. The function returns the next state and the reward obtained.

        i, j = state
        is_done= False

        if action == UP:
            next_state = [min(i + 1, WORLD_HEIGHT-1), min(j + 1, WORLD_WIDTH - 1)]
        elif action == DOWN:
            next_state = [max(i - 1, 0), min(j + 1, WORLD_WIDTH - 1)]
        elif action == ZERO:
            next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
        else:
            assert False
        
        # The reward is -1 for actions UP and DOWN. This is done to keep transactions to a minimum.
        reward = -1
        
        # ZERO gets a zero reward since we want to minimize the number of transactions
        if action == ZERO:
            reward = 0
        
        # Exceptions are 
        # i) If bankruptcy happens before WORLD_WIDTH time steps
        # ii) No deposit at initial state
        # iii) Redemption at initial state!
        # iv) Any action carried out from a bankrupt state
        if ((action == DOWN and i == 1 and 1 <= j < 10) or (
            action == ZERO and state == START) or (
            action == DOWN and state == START )) or (
            i == 0 and 1 <= j <= 10):    
                reward = -100
            
        # Next exception is when we get to the final time step.
        if (next_state[1] == WORLD_WIDTH - 1): 
            if (next_state[0] == 0): # Action resulted in ending with zero balance in final time step
                reward = 10
            else:
                reward = -10
            is_done = True       
        

        self.s = (WORLD_HEIGHT-1-next_state[0])*WORLD_WIDTH + next_state[1]    
        return next_state, reward, is_done

    def reset(self):
        self.s =36
        return 0
  
    def render(self, mode='human'):
        outfile = sys.stdout
        print(self.s)
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)

            #position[0]=WORLD_WIDTH - 1-position[0]
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')
