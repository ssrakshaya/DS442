import gymnasium as gym
from collections import defaultdict
import numpy as np
import random



#-------------
#Hyperparameters
#----------------
NUM_EPISODES = 100000      #Number of training episodes
ALPHA = 0.01               #Alpha - Learning rate (how much we update Q-values)
GAMMA = 0.99               #Gamma - Discount factor (how much we value future rewards)
EPSILON_START = 1.0        #Initial exploration rate
EPSILON_MIN = 0.01         #Minimum exploration rate
# DONT USE ITS OPTIONAL EPSILON_DECAY = 0.99995    # Decay rate per episode (optional)


#black jack has two actions, either you stick or hit
#for each state s, you need the Q value for hte state of either sticking or hitting
#so the code, For any new state, automatically initializes its two Q-values (hit/stick) to 0
Q = defaultdict(lambda: [0.0, 0.0])  # Q[state] = [Q(s,0), Q(s,1)]

#Helper Functions
def choose_action(state, epsilon): #pass in the state and exploration rate 

    #epsilon is what we use to determine exploration vs. exploitation 
    #with probability epsilon, you choose a random actions, which is exploration
    #with probability 1-epsilon, you choose the action with the highest Q value, which is exploitation
    """
    ε-greedy action selection.
    
    Args:
        state: Current game state (player_sum, dealer_card, usable_ace)
        epsilon: Exploration probability
    
    Returns:
        action: 0 (stick) or 1 (hit)
    """
    #random.random returns a decimal between 0-1. if that's less than epsilon, the agent ignores the Q table and picks a random action
    if random.random() < epsilon:
        #Exploration: random action
        return random.choice([0, 1])
    else:
        #Exploitation: choose action with highest Q-value
        if Q[state][0] >= Q[state][1]:
            return 0 #stick
        else:
            return 1 #hit - more cards







env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode= "human")
###YOUR Q-LEARNING CODE BEGINS





###YOUR Q-LEARNING CODE ENDS