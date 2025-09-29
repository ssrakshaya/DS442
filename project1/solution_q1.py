
def read_input(file_name = "input.txt"):
    with open(file_name, "r") as f:
        line = f.readline().strip()

    #format: M_left, C_left, M_right, C_right, Boat
    #where M_left and C_left are the number of missionaries and cannibals on the left bank,
    # M_right and C_right are the number of missionaries and cannibals on the right bank,
    # Boat is either L (left) or R (right).

    # Question 1.1.1a
    parts = line.split(",")
    M_left, C_left, M_right, C_right, Boat = parts
    return (int(M_left), int(C_left), int(M_right), int(C_right), Boat)

# def dfsRec(adj, visited, s, res):
#     #list marking what vertices have either been or not been visited 
#     visited[s] = True
#     res.append(s) 

#     #recursively visit 