from collections import deque

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
    #the state is good if not
    return True

def get_successors(state): 
    #technically the edges? so either there is one missionary 0 cannibal, 1 cannibal 0 missionary, 1 of each
    M_left, C_left, M_right, C_right, Boat = state
    """
    the paths are the diffeernt ways in which the missionaries and cannibals can move
    the  boat can carry 1 or 2 people. The valid moves are:
    1 Missionary (M)
    1 Cannibal (C)
    2 Missionaries (M, M)
    2 Cannibals (C, C)
    1 Missionary and 1 Cannibal (M, C)
    """
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

def dfs(start):

    """
    DFS is a graph traversal algorithm that explores the deepest path possible along a branch before backtracking
    in the context of the missionaries and cannibals:
    - the different successor states are the nodes which are entered into stack
    - dfs starts at the first state (given by input.txt) and moved in all the posssible directions (with the for loop it
    iterating through all possible moves) going further down the tree
    - finally, whenever the dfs algorithm reaches either the goal state of (0,0,3,3,R) or a dead end, 
    it backtracks, and explores other paths
    """
    #the stack has bot the current state, and the path
    stack = [(start, [start])]
    visited = set() #making a set for visited that is empty because nothing has been visited yet
    explored = 0 #this counts the number of paths which had been explored previously

    while len(stack) > 0: 
        state, path = stack.pop() #state is the current node, while path is the sequence of nodes which make up DFS
        if state in visited: #if the node has already been vsited, skip to next
            continue

        # visited.add(state) #mark the current state as visited
        # explored += 1 #you have explored further along the tree

        if is_goal(state): #if the state is equivalent to the goal state
            return path, len(path)-1, explored
        
        visited.add(state) #mark the current state as visited
        explored += 1 #you have explored further along the tree
        
        for succ in get_successors(state):
            if succ not in visited:
                stack.append((succ, path + [succ]))
        
    return None, None, explored

def bfs(start):
    """
    Breadth First: finds the shortest and fewest action path
    BFS explores a graph or tree level by level. BFS visits ALL its immediate neighbors before moving on
    to the next level of nodes. This ensures all nodes at one level are processed 
    """

    #the stack has both the current state, and the path
    queue = deque([(start, [start])])
    visited = set() #making a set for visited that is empty because nothing has been visited yet
    expansions = 0 #this counts the number of paths which had been explored previously

    while queue: #while there are still nodes to process
        state, path = queue.popleft() #BFS uses a QUEUE NOT STACK - first in first out
        #expansions +=1 #add cost to the exploration

        if state in visited: #check if node has been visited
            continue
        
        # visited.add(state) #mark node as visited
        # expansions +=1 #add cost to the exploration

        ## Goal check BEFORE expansion if you want to NOT count the goal as an expansion.
        if is_goal(state):
            return path, len(path)-1, expansions #returning the matching node path
        
        visited.add(state) #mark node as visited
        expansions +=1 #add cost to the exploration

        #enqueue all univisted neighbors (perhaps children) of the current node
        for succ in get_successors(state):
            if succ not in visited:
                queue.append((succ, path + [succ])) #adding unvisted neighbors to the quueue
        
    return None, None, expansions





if __name__ == "__main__" : 

    start = read_input("input.txt")  # Read initial state from input.txt
    #print(start)

    # Run DFS to solve the puzzle
    path, cost, expansions = dfs(start)
     # Print the solution in the required format
    print("The solution of Q1.1.a (DFS) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)

    #goal state: (0, 0, 3, 3, R)

    #Run BFS:
    path, cost, expansions = bfs(start)
    print("The solution of Q1.1.b (BFS) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)

    




