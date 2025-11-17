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
    Arguments:
        state: Current game state (player_sum, dealer_card, usable_ace)
        epsilon: Exploration probability (are we exploring or exploiting)
    
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

def update_q(state, action, reward, next_state, terminated):
    """
    This function updated one entry in the Q-table, with the value of taing an 'action' in the 'state' 
    Q-learning update rule.
    
    Q(s,a) ← Q(s,a) + alpha[r + y·max_a'Q(s',a') - Q(s,a)]

    Look at old Q-value for (state, action).
    Use the reward + future value to compute a better estimate.
    Slowly adjust Q(s,a) toward that better estimate.

    Stop considering future value if the episode has ended.

    td is temporal difference looking at teh difference vetween the old estimate of the action value at time to Q(s,a)
    and the new better estimate using time t+1 
    """
    #Get the current estimate Q-value Q(s,a)
    current_q = Q[state][action] #-> this is the agent's previous belief on how good it is to take an action in a state 
    
    #If episode terminated, there's no next state (max Q = 0)
    if terminated: #handling terminal states 
        max_next_q = 0.0 #if the epsiode is over, there is no future reward, and the value of the next state is 0 
    else: #otherwise computing the future value of the state 
        #Max Q-value for next state (best action we could take)
        max_next_q = max(Q[next_state])
    
    #TD target: r + γ·max_a'Q(s',a')
    td_target = reward + GAMMA * max_next_q #this is the number we want the current Q value to become
    
    #TD error: difference between target and current estimate
    td_error = td_target - current_q
    
    #Update Q-value
    Q[state][action] = current_q + ALPHA * td_error








env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode= "human")
###YOUR Q-LEARNING CODE BEGINS





###YOUR Q-LEARNING CODE ENDS