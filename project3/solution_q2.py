import gymnasium as gym
import numpy as np
from collections import defaultdict

# Hyperparameters
NUM_EPISODES = 1000
GAMMA = 0.99  # Discount factor
THETA = 1e-8  # Convergence threshold for value iteration

# Step 1: Initialize environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", 
               is_slippery=True, render_mode=None)

# Step 2: Random exploration (Q2.2)
# Step 3: Learn MDP model (Q2.2)
# Step 4: Value iteration (Q2.3)
# Step 5: Policy extraction (Q2.4)
# Step 6: Execute optimal policy (Q2.5)

def collect_random_exploration_data(env, num_episodes):
    """
    Execute random policy for num_episodes and collect transitions.
    
    Returns:
        transitions: list of (state, action, next_state, reward) tuples
    """
    transitions = [] #transitions are lists that have state, the action, the next state and the reward in them 
    
    print(f"Collecting data from {num_episodes} random episodes...")
    
    #repeating the proccess for how many episodes it needs to be done
    for episode in range(num_episodes):
        state, info = env.reset() #env.reset() starts a new episode and returns the starting state.
        state = int(state)  #Convert to int
        terminated = False #did we reach a terminal state, either a hole or goal
        truncated = False #truncated means the episode ended because of max steps 
        
        while not (terminated or truncated): #continue taking actions until you hit a terminal state
            #this is the Random policy action - exploration. 
            action = env.action_space.sample()
            
            #Next state is the state the environment transitions to, reward is 0. terminated means the episode ended normally
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state) #convert the next_stat to an integer
            
            #Store transition and record it 
            #(s, a, s', r). this will help for calcaulting the transition probability and 
            transitions.append((state, action, next_state, reward))
            
            #Move to next state
            state = next_state
        
        #printing progress
        if (episode + 1) % 100 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    print(f"Collected {len(transitions)} transitions\n")
    return transitions


#calculates the MDP model values
def estimate_mdp_model(transitions, num_states, num_actions):
    """
    Estimate the probabilities of a certain transition ocurring, and the rewards that result from it
    
    Arguments: 
        transitions: list of (state, action, next_state, reward)
        num_states: number of states (16 for 4x4 grid)
        num_actions: number of actions (4)
    
    Returns:
        T: transition probabilities [num_states, num_actions, num_states]
        R: rewards [num_states, num_actions, num_states]
    """
    #Count the number of transitons, so the probabiltity fo a certain transition ocurring
    count_sa_s = defaultdict(int)  #count[(s, a, s')] counts how many times we observed this exact transition
    count_sa = defaultdict(int) #count[(s, a)] counts how many total times we took action a in state s
    
    #The total rewards you get from a specific transiio
    sum_reward = defaultdict(float)  # sum[(s, a, s')] = total reward
    
    print("Estimating MDP model from transitions...")
    
    #iterating through all of the different transitions 
    for (s, a, s_next, r) in transitions:
        count_sa_s[(s, a, s_next)] += 1 #increment the s,a,s' transition count
        count_sa[(s, a)] += 1 #increment the s,a count
        sum_reward[(s, a, s_next)] += r #add the reward to the total 
    
    #Initialize T and R arrays, creating empty matrices 
    T = np.zeros((num_states, num_actions, num_states)) #T[s,a,s']--> probability of transitioning to s'
    R = np.zeros((num_states, num_actions, num_states)) #R[s,a,s'] --> expected reward of going to state s' after going from state s and taking action a
    
    # Compute T(s,a,s') = count(s,a,s') / count(s,a)
    # Compute R(s,a,s') = sum_reward(s,a,s') / count(s,a,s')
    for (s, a, s_next), count in count_sa_s.items():
        T[s, a, s_next] = count / count_sa[(s, a)]
        R[s, a, s_next] = sum_reward[(s, a, s_next)] / count
    
    print("MDP model estimation complete!\n")
    return T, R