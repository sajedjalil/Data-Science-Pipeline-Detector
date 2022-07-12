# copy paste the following functions into your script or notebook
def load_event(filename):
    hits = pd.read_csv(filename+'-hits.csv')
    cells = pd.read_csv(filename+'-cells.csv')
    particles = pd.read_csv(filename+'-particles.csv')
    truth = pd.read_csv(filename+'-truth.csv')
    return hits, cells, particles, truth