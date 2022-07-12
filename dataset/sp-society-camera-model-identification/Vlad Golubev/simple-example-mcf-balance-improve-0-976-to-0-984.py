from ortools.graph import pywrapgraph
def mcf_balance(df):
#     ['HTC-1-M7', 'iPhone-6', 'Motorola-Droid-Maxx', 'Motorola-X','Samsung-Galaxy-S4', 
#      'iPhone-4s', 'LG-Nexus-5x', 'Motorola-Nexus-6','Samsung-Galaxy-Note3', 'Sony-NEX-7']
    cols = df.columns[2:12] # columns with values prob
    df[cols] *= 1000000000
    df[cols].astype(np.int64)
    size = df.shape[0]
    mcf = pywrapgraph.SimpleMinCostFlow()
    m = df[cols].as_matrix()
    for j in range(10):
        mcf.SetNodeSupply(j+size, size//10)
    for i in range(size):
        mcf.SetNodeSupply(i, -1)
        for j in range(10):
            mcf.AddArcWithCapacityAndUnitCost(j+size, i, 1, int(-m[i][j]))
    mcf.SolveMaxFlowWithMinCost()

    answ = np.zeros(size, dtype=np.int32)
    for i in range(mcf.NumArcs()):
        if mcf.Flow(i) > 0:
            answ[mcf.Head(i)] = mcf.Tail(i) - size
    df['camera'] = cols[answ]

# manip = df[df['fname'].str.contains('manip')]
# unalt = df[df['fname'].str.contains('unalt')]
# mcf_balance(manip)
# mcf_balance(unalt)