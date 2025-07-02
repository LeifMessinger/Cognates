def conductance(arr, adj):
    outside = 0
    inside = 0

    for i in arr:
        for j in range(len(adj)):

            if adj[i][j] == -1:
                continue

            if j in arr:
                inside += adj[i][j] / 4
            else:
                outside += adj[i][j]

    if outside == 0 or inside == 0:
        return -1

    # 1 - because of the inverse
    return 1 - outside / ((2 * inside) + outside)

def density(arr, adj):
    a = 0

    for i in arr:
        for j in arr:
            a += adj[i][j]

    return a / (len(arr) * len(arr))