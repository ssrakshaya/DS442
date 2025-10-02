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