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
    
    #Compute T(s,a,s') = count(s,a,s') / count(s,a)
    #Compute R(s,a,s') = sum_reward(s,a,s') / count(s,a,s')
    for (s, a, s_next), count in count_sa_s.items():
        T[s, a, s_next] = count / count_sa[(s, a)]
        R[s, a, s_next] = sum_reward[(s, a, s_next)] / count
    
    print("MDP model estimation complete!\n")
    return T, R


def value_iteration(T, R, num_states, num_actions, gamma=0.99, theta=1e-8):
    """
    Perform value iteration to find optimal value function.
    
    Bellman optimality equation:
    V(s) = max_a Σ_{s'} T(s,a,s') [R(s,a,s') + γ·V(s')]
    
    Args:
        T: transition probabilities [num_states, num_actions, num_states]
        R: rewards [num_states, num_actions, num_states]
        gamma: discount factor
        theta: convergence threshold
    
    Returns:
        V: optimal value function [num_states]
    """
    #Initialize all state valyes to zero
    V = np.zeros(num_states)
    
    print("Running Value Iteration...")
    iteration = 0
    
    #main value iteration will continue until V(s) stops changing significantly
    while True:
        delta = 0  #Tracks the maximum change in value function across states
        
        #Update each state's value
        for s in range(num_states):
            v_old = V[s] #store the old value so that you can find the change 
            
            # Compute value for each action
            action_values = [] #create a list to store the Q(s,a) for each action a
            for a in range(num_actions):
                # Q(s,a) = Σ_{s'} T(s,a,s') [R(s,a,s') + γ·V(s')]
                q_value = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V)) #(R + gamma * V) is a vector over all next states s'
                action_values.append(q_value)
            
            #V(s) = max_a Q(s,a)
            #choose the action that gives the max Q value
            V[s] = max(action_values)
            
            #Update delta to measure the largest value change
            delta = max(delta, abs(v_old - V[s]))
        
        iteration += 1
        
        #Convergence check — if improvement is tiny, we stop
        if delta < theta:
            print(f"Value Iteration converged after {iteration} iterations\n")
            break
        #Print debugging info every 10 iterations
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: delta = {delta:.6f}")
    
    return V


def extract_policy(V, T, R, num_states, num_actions, gamma=0.99):
    """
    Extract optimal policy from value function.
    
    Formula Used: π*(s) = argmax_a Σ_{s'} T(s,a,s') [R(s,a,s') + γ·V(s')]
    
    Arguments:
        V: optimal value function
        T: transition probabilities
        R: rewards
        num_states -> number of states (16 for the frozen lake)
        num_action -> number of actions which can be taken which is 4
    
    Returns:
        policy: array of length num_states where policy[s] = best action to take in state s
    """

    #initialize the policy, one action is given per state
    policy = np.zeros(num_states, dtype=int)
    
    print("Extracting optimal policy...")
    
    for s in range(num_states): #loop through all the states to choose the best action in each 
        action_values = [] #stores the Q(s,a) value for each action
        
        #Compute Q(s,a) for every possible action a
        for a in range(num_actions):
            # Q(s,a) = Σ_{s'} T(s,a,s') [R(s,a,s') + γ·V(s')]
            q_value = np.sum(T[s, a, :] * (R[s, a, :] + gamma * V))
            action_values.append(q_value)
        
        #Choose action with highest Q-value
        policy[s] = np.argmax(action_values)
    
    print("Policy extraction complete!\n")
    return policy


if __name__ == "__main__":
    # Initialize environment (without rendering for training)
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", 
                   is_slippery=True, render_mode=None)
    
    # Q2.2: Collect data and learn MDP model
    transitions = collect_random_exploration_data(env, NUM_EPISODES)
    T, R = estimate_mdp_model(transitions, NUM_STATES, NUM_ACTIONS)
    
    # Q2.3: Value Iteration
    V = value_iteration(T, R, NUM_STATES, NUM_ACTIONS, GAMMA, THETA)
    
    # Q2.4: Policy Extraction
    policy = extract_policy(V, T, R, NUM_STATES, NUM_ACTIONS, GAMMA)
    
    # Close training environment
    env.close()
    
    # Q2.5: Execute optimal policy with rendering
    execute_optimal_policy(policy, NUM_TEST_EPISODES)