import heapq
from math import inf
import math

def read_input(file_name = "input.txt"):

    #reeading initial file from input.tct
    with open(file_name, "r") as f:
        line = f.readline().strip()

    #format: M_left, C_left, M_right, C_right, Boat
    #where M_left and C_left are the number of missionaries and cannibals on the left bank,
    # M_right and C_right are the number of missionaries and cannibals on the right bank,
    # Boat is either L (left) or R (right).

    # Question 1.1.1a
    parts = [p.strip() for p in line.split(",")]
    M_left, C_left, M_right, C_right, Boat = parts
    return (int(M_left), int(C_left), int(M_right), int(C_right), Boat)


def is_goal(state):
    #checking if goal state has been reached (all missionaries and cannibals are on the right side)
    M_left, C_left, M_right, C_right, Boat = state 
    return M_left == 0 and C_left == 0 #checking that missionaries and cannibals are both zero
    #everyone involved has moved to the right bank


def is_valid(state):
    """
    Making sure all the state values make sense, so they arent negative, or 
    there cannibals outnumbering missionaries, 
    This method just makes sure the number of missionaries to cannibals is proper before moving ahead with the next successor state
    """

    M_left, C_left, M_right, C_right, Boat = state
    # No negative values
    if min(M_left, C_left, M_right, C_right) < 0:
        return False
    # If missionaries are on the left, cannibals can't outnumber them
    if M_left > 0 and C_left > M_left:
        return False
    # If missionaries are on the right, there cannot be more cannibals than missionaries
    if M_right > 0 and C_right > M_right:
        return False
    #the state is good if none of the above conditions are ocurring 
    return True

def get_successors(state):
    """
    this method helps find the next states, which are possible and can be sequenced
    next after the current state.

    the paths are the different ways in which the missionaries and cannibals can move
    the  boat can carry 1 or 2 people. The valid moves are:
    1 Missionary (M)
    1 Cannibal (C)
    2 Missionaries (M, M)
    2 Cannibals (C, C)
    1 Missionary and 1 Cannibal (M, C)
    """
    M_left, C_left, M_right, C_right, Boat = state
    paths = [(0,1), (1,0), (1, 1), (2,0), (0, 2)]
    successor = []

    if Boat == "L":
        for m, c in paths:
            #the different possibilites of the new state, are looped through by the for loop
            new_state = (M_left - m, C_left - c, M_right + m, C_right + c, "R")
            if is_valid(new_state):
                successor.append(new_state) #adding the next state to the path (the state is acting like a node in a tree)
    else: #if the boat is on the right side, you would move in the opposing direction
        for m, c in paths:
            new_state = (M_left + m, C_left + c, M_right - m, C_right - c, "L")
            if is_valid(new_state):
                successor.append(new_state) #adding this state on to the tree path
    
    return successor


#calculates the cost of each step
def step_cost(current, next):
    """
    Finding costs for both Cost model A and cost model B
    Cost Model A:
    - cost = 2 for each missionary
    - cost = 1 for each cannibal

    Cost Model A: cost by direction the boat is traveling in
    - L->R costs 2
    - R->L costs 1
    """

    M_left, C_left, M_right, C_right, Boat  = current
    M_left2, C_left2, M_right2, C_right2, Boat2 = next #the next state that can be iterated to 

    if Boat == "L": #the boat is moving fomr left to right (upstream, costs more)
        moved_M = M_left - M_left2 #the number of missionaries moved are calcualted from state x to state x+1
        moved_C = C_left - C_left2 #moved cannibals are calculated from preivious to next step
    else:
        moved_M = M_right - M_right2 #the number of missionaries moved are calcualted from state x to state x+1
        moved_C = C_right - C_right2 #moved cannibals are calculated from preivious to next step
    
    return 2*moved_M + moved_C

#---------------- Heuristics for A* ----------------------
def h1(state):
    #Heuristic 1, adding the passengers weights
    M_left, C_left, M_right, C_right, Boat  = state
    h1_total = (2*M_left) + (C_left)
    return h1_total

def h2(state):
    M_left, C_left, M_right, C_right, Boat  = state
    h2_total = math.ceil((2*M_left + C_left) / 3)
    return h2_total

def h3(state):
    M_left, C_left, M_right, C_right, Boat = state
    return max(2*M_left, C_left)


def reconstruct_path(best_g, start, goal):
    #reconstructing the path from start goal to end goal by following the nodes that have been visited so far
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = best_g[current][1]
    
    path.reverse()
    return path

#---------------A star search using cost model A---------------
def astar(start, heuristic):
    """
    A* search with Cost Model A
    What A* does is add a priority queue with an estiameed cost
    Priority Queue Cost = f = g +h(state), where g gets the accumulated cost added to it
    this method returns the solution path, how much it cost, and the number of node expansions
    """

    #initialize the starting node, the priority queue constains the 
    #cost of starting state (f), g cost (at 0 currently ), and the state
    pq = [(heuristic(start), 0, start)]
    #the best g cost contains the best g cost and the parent state it traveled from 
    best_g = {start: (0, None)}
    expansions = 0

    while pq:
        #get node with lowest f value
        f, g, state = heapq.heappop(pq)

        #skipping entries if there is already a cheaper known g value for it
        if g != best_g.get(state, (inf, None))[0]:
            continue

        #before adding on a node expansion count, check whether it matches the goal state
        if is_goal(state):
            path = reconstruct_path(best_g, start, state)
            return path, g, expansions
        
        expansions += 1

        #for loop goes through all the neighbors of the current state
        for succ in get_successors(state):
            #computing the cost of the step by passing into the cost calculating function for model A
            g_step = step_cost(state, succ)
            new_g = g + g_step #g is cumulative cost to the current state
            
            #if this successor has not been visited, or it has a cost lower than another one, add it
            if succ not in best_g or new_g < best_g[succ][0]:
                best_g[succ] = (new_g, state) #record the new path
                new_f = new_g + heuristic(succ)
                heapq.heappush(pq, (new_f, new_g, succ)) #push the neighbors into the priority queue

    return None, None, expansions




if __name__ == "__main__" : 

    start = read_input("input.txt")  # Read initial state from input.txt
    #print(start)

    # Run H1 with A*
    #path, cost, expansions = dfs(start)
    # Print the solution in the required format
    # Model A
    path, cost, expansions = astar(start, h1)
    print("The solution of Q3.1 (Heuristic 1) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)

    #goal state: (0, 0, 3, 3, R)

    #Run H2 with A*
    path, cost, expansions = astar(start, h2)
    print("The solution of Q3.1 (Heuristic 2) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)

    #Run H3 with A*
    # Heuristic 3
    path, cost, expansions = astar(start, h3)
    print("The solution of Q3.1 (Heuristic 3) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)
