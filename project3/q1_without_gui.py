import gymnasium as gym
from collections import defaultdict
import numpy as np
import random



#-------------
#Hyperparameters
#----------------
#does not use the gui
NUM_EPISODES = 100000      #Number of training episodes
ALPHA = 0.01               #Alpha - Learning rate (how much we update Q-values)
GAMMA = 0.99               #Gamma - Discount factor (how much we value future rewards)
EPSILON_START = 1.0        #Initial exploration rate
EPSILON_MIN = 0.01         #Minimum exploration rate
## DONT USE ITS OPTIONAL 
EPSILON_DECAY = 0.99995    # Decay rate per episode


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


def train_blackjack():
    """
    Train Q-learning agent on Blackjack and track performance.
    - This plays many games of Blackjack using Q-learning, and each game slightly improves the Q-table's
    Q value so that the agent gets better


    """
    #create environment
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=None)
    #TAKE OUT REMOVE BAD
    # Note: Use render_mode=None for training (much faster!)
    # Use render_mode="human" only for final visualization
    
    #Points tracking
    #how well is t he agent doing? 
    wins = 0
    losses = 0
    draws = 0
    epsilon = EPSILON_START
    
    print("Starting Q-Learning Training for Blackjack: ")
    print(f"Episodes: {NUM_EPISODES}, α={ALPHA}, γ={GAMMA}, ε={EPSILON_START}→{EPSILON_MIN}\n")
    
    for episode in range(1, NUM_EPISODES + 1): #how many rounds of training
        #Reset environment and get initial state
        state, info = env.reset() #env.reset starts a few blackjack game and gives the first state
        terminated = False #is game over
        truncated = False #is game over
        episode_reward = 0 #reward is either -1, 0, or +1

        #Play one episode
        while not (terminated or truncated): #while not means keep going until the game ends
            #Choose action using ε-greedy policy
            action = choose_action(state, epsilon)
            
            # Take action and observe result
            next_state, reward, terminated, truncated, info = env.step(action) #env.step applies the action to the environment
            #the next state: new situation after the move
            #reward: 0 during the game, but -1,0, or +1 at the end 
            #terminated tell u whether the game has ended or not
            
            #Update Q-table, given the previous state and action, change the estiamte of how good this move was (training the q learning)
            update_q(state, action, reward, next_state, terminated or truncated)
            
            #Move to next state
            state = next_state
            episode_reward += reward #dding rewards 
        
        # Track results (reward is +1, 0, or -1)
        if episode_reward > 0:
            wins += 1
            result = "WIN"
        elif episode_reward < 0: #lost
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        #Decaying epsilon (reducing exploration over time)
        #when beginning training, the epslon vaue is large because of random exploration
        #but over time, u shrink towards the minimum epsiolon, so you rely more on the epsilon and not just random guessing
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        #Print progress every 1000 episodes
        if episode % 1000 == 0:
            total_games = wins + losses + draws
            win_rate = (wins / total_games) * 100 if total_games > 0 else 0
            print(f"Episode {episode:6d} | {result:4s} | "
                  f"Win Rate: {win_rate:.2f}% | "
                  f"W/L/D: {wins}/{losses}/{draws} | "
                  f"ε: {epsilon:.4f}")
    
    env.close() #shuts down the environment.

    #Final scores and values
    total_games = wins + losses + draws
    win_rate = (wins / total_games) * 100
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total Episodes: {NUM_EPISODES}")
    print(f"Wins:   {wins:6d} ({(wins/total_games)*100:.2f}%)")
    print(f"Losses: {losses:6d} ({(losses/total_games)*100:.2f}%)")
    print(f"Draws:  {draws:6d} ({(draws/total_games)*100:.2f}%)")
    print(f"Final Win Rate: {win_rate:.2f}%")
    print("="*60)
    
    return Q  #Return trained Q-table



# ============================================================
# OPTIONAL: TEST TRAINED AGENT
# ============================================================
def test_agent(Q_table, num_test_episodes=1000):
    """
    Test the trained agent without exploration (pure exploitation).
    """
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=None)
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nTesting trained agent for {num_test_episodes} episodes...")
    
    for episode in range(num_test_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Always choose best action (no exploration)
            action = 0 if Q_table[state][0] >= Q_table[state][1] else 1
            state, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    
    env.close()

    test_win_rate = (wins / num_test_episodes) * 100
    print(f"Test Results: W/L/D = {wins}/{losses}/{draws}")
    print(f"Test Win Rate: {test_win_rate:.2f}%\n")





if __name__ == "__main__":
    # Train the agent
    trained_Q = train_blackjack()
    
    # Optional: Test the trained agent
    test_agent(trained_Q, num_test_episodes=1000)



###YOUR Q-LEARNING CODE ENDS