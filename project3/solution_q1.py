import gymnasium as gym
from collections import defaultdict
import random



#-------------
#Hyperparameters
#----------------
#does not use the gui
#NUM_EPISODES = 100000      #Number of training episodes

#FOR RENDERING VS. NOT RENDERING
NUM_TRAINING_EPISODES = 100000  # Training episodes (no rendering)
NUM_DEMO_EPISODES = 10          # Demo episodes (with rendering)

ALPHA = 0.01               #Alpha - Learning rate (how much we update Q-values)
GAMMA = 0.99               #Gamma - Discount factor (how much we value future rewards)
EPSILON_START = 1.0        #Initial exploration rate
EPSILON_MIN = 0.01         #Minimum exploration rate
EPSILON_DECAY = 0.99995    #Decay rate per episode


#black jack has two actions, either you stick or hit
#for each state s, you need the Q value for hte state of either sticking or hitting
#so the code, For any new state, automatically initializes its two Q-values (hit/stick) to 0
Q = defaultdict(lambda: [0.0, 0.0])  #Q[state] = [Q(s,0), Q(s,1)]

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

    td is temporal difference looking at the difference vetween the old estimate of the action value at time to Q(s,a)
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




#VERSION TWO OF TRAINING BLACK JACK WITOUT RENDERING (SO INCREASED SPEED)
def train_blackjack(): #This function will run Q-learning for many Blackjack games, also called as episodes
    #training without rending in the GUI so that its faster
    
    #Train WITHOUT rendering for speed
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode=None) 
    #env = gym.make(...) creates the Blackjack environment from Gymnasium.

    #nitialize counters for tracking performance of the agent 
    wins = 0
    losses = 0
    draws = 0
    epsilon = EPSILON_START #epsilon --> exploration rate, means the probability of taking a random action at the beginning of playing
    
    print("-------------------------------------------------------------------------")
    print("TRAINING PHASE (No Rendering)")
    print("------------------------------------------------------------------------")
    print(f"Training for {NUM_TRAINING_EPISODES} episodes...\n")
    
    #main training loop for each game of black jack, looping from 1 to the number of training episodes
    for episode in range(1, NUM_TRAINING_EPISODES + 1): #each episode represents one complete game of blackjack
        state, info = env.reset() #starts a new black jack game, and returns the intial state
        terminated = False #ddi the game end with a win/loss/bust
        truncated = False #game ends for a differetn reason
        episode_reward = 0 #+1, -1, or 0 based on win, loss, draw
    
        #while the game is not over, keep playing
        while not (terminated or truncated):
            action = choose_action(state, epsilon) #chooses either a random action (for exploration) or uses the best Q value action (exploitation)

            #next state is the new game state after the action
            #reward, is the immediate reward (usually zero till the game ends)
            #terminate and truncate are whether the game has ended
            next_state, reward, terminated, truncated, info = env.step(action)

            #This calls the Q-learning update function: Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]
            #adjusting an estimate of how good an action is
            update_q(state, action, reward, next_state, terminated or truncated)
            state = next_state #next state is set to be current
            episode_reward += reward 
        
        #Track episode rewards
        #if game has ended, update who has won, lost, or drawed 
        if episode_reward > 0:
            wins += 1
            result = "WIN"
        elif episode_reward < 0:
            losses += 1
            result = "LOSS"
        else: 
            draws += 1
            result = "DRAW"
        
        #Decay epsilon, meaning with each round, explore less and less (less exploring more exploitation)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        #Print progress every 1000 episodes
        if episode % 1000 == 0: 
            total_games = wins + losses + draws
            win_rate = (wins / total_games) * 100
            print(f"Episode {episode:6d} | {result:4s} | Win Rate: {win_rate:.2f}%")
    
    env.close() #shuts down the Blackjack environment.

    # Training summary
    total_games = wins + losses + draws
    win_rate = (wins / total_games) * 100
    
    print("\n" + "----------------------------------------------------------------")
    print("TRAINING COMPLETE!")
    print("-----------------------------------------------------------------------")
    print(f"Total Episodes: {NUM_TRAINING_EPISODES}")
    print(f"Wins:   {wins:6d} ({(wins/total_games)*100:.2f}%)")
    print(f"Losses: {losses:6d} ({(losses/total_games)*100:.2f}%)")
    print(f"Draws:  {draws:6d} ({(draws/total_games)*100:.2f}%)")
    print(f"Final Win Rate: {win_rate:.2f}%")
    print("-----------------------------------------------------------------------" + "\n")


# ============================================================
# DEMONSTRATION PHASE (WITH RENDERING)
# ============================================================

def demonstrate_learned_policy():
    #demonstrate learned policy render_mode set to human 
    
    #NOW use render_mode="human" as per instructions
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
    
    wins = 0
    losses = 0
    draws = 0
    
    print("-----------------------------------------------------------------------")
    print("DEMONSTRATION PHASE (With Rendering)")
    print("-----------------------------------------------------------------------")
    print(f"Playing {NUM_DEMO_EPISODES} episodes with trained policy...")
    print("GUI window will show the game!\n")
    
    #Loop through demo episodes
    for episode in range(1, NUM_DEMO_EPISODES + 1):
        state, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        #Play using learned policy (no exploration, only pure exploitation)
        while not (terminated or truncated):
            #Choose best action according to Q-table
            if Q[state][0] >= Q[state][1]:
                action = 0
            else:
                action = 1
            
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Track results
        if episode_reward > 0:
            wins += 1
            result = "WIN"
        elif episode_reward < 0:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"
        
        #Print each episode result
        total_games = wins + losses + draws
        win_rate = (wins / total_games) * 100
        print(f"Episode {episode}: {result} | Win Rate: {win_rate:.2f}%")
    
    env.close()

    print("\n" + "----------------------------------------------------------------")
    print("DEMONSTRATION COMPLETE!")
    print("----------------------------------------------------------------")
    print(f"Demo Episodes: {NUM_DEMO_EPISODES}")
    print(f"Wins:   {wins} ({(wins/NUM_DEMO_EPISODES)*100:.2f}%)")
    print(f"Losses: {losses} ({(losses/NUM_DEMO_EPISODES)*100:.2f}%)")
    print(f"Draws:  {draws} ({(draws/NUM_DEMO_EPISODES)*100:.2f}%)")
    print(f"Win Rate: {win_rate:.2f}%")
    print("----------------------------------------------------------------")





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
    # # Train the agent
    # trained_Q = train_blackjack()
    
    # # Optional: Test the trained agent
    # test_agent(trained_Q, num_test_episodes=1000)

    # Phase 1: Train the agent (fast, no rendering)
    train_blackjack()
    
    # Phase 2: Demonstrate with rendering (as per instructions)
    demonstrate_learned_policy()


