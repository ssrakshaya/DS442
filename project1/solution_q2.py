import heapq
from math import inf

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
def step_cost(current, next, model):
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
        direction = "LR" #setting direction
    else:
        moved_M = M_right - M_right2 #the number of missionaries moved are calcualted from state x to state x+1
        moved_C = C_right - C_right2 #moved cannibals are calculated from preivious to next step
        direction = "RL" #setting direction

    if model == "A":
        return 2*moved_M + 1*moved_C #returning cost of carrying Missionaries and cannibals in cost plan A
    else:
        if direction =="LR":
            return 2
        else:
            return 1


def reconstruct_path(visited, start, goal):
    path = [goal]
    current = goal
    while current != start:
        current = visited[current][1]
        path.append(current)
    
    path.reverse()
    return path

def ucs(start, model):
    pq = [(0, start)]
    visited = {start: (0, None)}        # state -> (best_g, parent)
    expansions = 0

    while pq:
        g, state = heapq.heappop(pq)

        # skip stale entries
        if g != visited.get(state, (inf, None))[0]:
            continue

        if is_goal(state):
            path = reconstruct_path(visited, start, state)
            return path, g, expansions

        expansions += 1

        for succ in get_successors(state):
            step = step_cost(state, succ, model)
            new_g = g + step
            if succ not in visited or new_g < visited[succ][0]:
                visited[succ] = (new_g, state)
                heapq.heappush(pq, (new_g, succ))

    return None, None, expansions

#HELPER FUNCTION
def show_actions_with_cost(path, model):
    if not path:
        print("No solution path.")
        return
    total = 0
    print("Actions (with step costs):")
    for prev, nxt in zip(path, path[1:]):
        # derive moved passengers + direction
        M_left, C_left, M_right, C_right, Boat  = prev
        M_left2, C_left2, M_right2, C_right2, Boat2 = nxt
        if Boat == 'L':
            moved_M, moved_C, direction = (M_left - M_left2), (C_left - C_left2), "L→R"
        else:
            moved_M, moved_C, direction = (M_right - M_right2), (C_right - C_right2), "R→L"

        step = step_cost(prev, nxt, model)
        total += step
        print(f"  {direction}: {moved_M}M, {moved_C}C   cost = {step}")
    print(f"Recomputed total cost = {total}\n")



#CHAT 
"""
def uniform_cost(start, model):
    
    # starting state has zero code. then remove the next node from the list, which is currently technically the node with the minimum cost
    # if selected path is the target, it can be returning
    # when the boat = L, then we know the missionaries and cannibals are moving from left to right
    # when boat = r M and C or moving right to left
    # when computing the path (so next state) which is traveled, you also need to add on the 
    # the cost of each missionary nad each cannibal

    

    heap = [(0, start, [start])]   # (g cost, state, path following)
    best_g = {start: 0} #the best cost possible thus far with the path we go through
    expansions = 0

    while heap:
        #g is cost so far to reach a state, 
        #state is the current tuple of locations where cannibals missionaries and everyone else are in
        #path, tuple seuences used to get there 
        g, state, path = heapq.heappop(heap)

        if g!= best_g.get(state, inf):
            continue
            
        # Goal test BEFORE expansion → don't count goal as expanded
        if is_goal(state):
            return path, g, expansions

        expansions += 1

        for succ in get_successors(state):
            step = step_cost(state, succ, model)
            new_g = g + step
            if new_g < best_g.get(succ, inf):
                best_g[succ] = new_g
                heapq.heappush(heap, (new_g, succ, path + [succ]))

    return None, None, expansions
"""







if __name__ == "__main__" : 

    start = read_input("input.txt")  # Read initial state from input.txt
    #print(start)

    # Run Cost Model A with UCS
    #path, cost, expansions = dfs(start)
    # Print the solution in the required format
    # Model A
    path, cost, expansions = ucs(start, "A")
    print("The solution of Q2.1 (UCS, cost model A) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)
    show_actions_with_cost(path, "A")   # <— add this

    #goal state: (0, 0, 3, 3, R)

    #Run Cost Model B with UCS:
    path, cost, expansions = ucs(start, "B")
    print("The solution of Q2.1 (UCS, cost model B) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)
    show_actions_with_cost(path, "B")