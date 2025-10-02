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





def uniform_cost():
    """
    starting state has zero code. then remove the next node from the list, which is currently technically the node with the minimum cost
    if selected path is the target, it can be returning
    when the boat = L, then we know the missionaries and cannibals are moving from left to right
    when boat = r M and C or moving right to left
    when computing the path (so next state) which is traveled, you also need to add on the 
    the cost of each missionary nad each cannibal

    """









if __name__ == "__main__" : 

    start = read_input("input.txt")  # Read initial state from input.txt
    #print(start)

    # Run Cost Model A with UCS
    #path, cost, expansions = dfs(start)
     # Print the solution in the required format
    print("The solution of Q2.1 (UCS, cost model A) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)

    #goal state: (0, 0, 3, 3, R)

    #Run BFS:
    #path, cost, expansions = bfs(start)
    print("The solution of Q2.1 (UCS, cost model B) is:")
    print("Solution Path:", path)
    print("Total cost =", cost)
    print("Number of node expansions =", expansions)