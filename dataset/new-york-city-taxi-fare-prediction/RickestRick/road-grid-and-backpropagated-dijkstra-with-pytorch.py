import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.optim.lr_scheduler import CosineAnnealingLR

import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.sparse.csgraph import minimum_spanning_tree
import pylab as pl
from matplotlib import collections  as mc
import os

print(os.listdir("../input"))

start_time = time.time()
inp = "../input/train.csv"
gpu = -1

#############################################################################
#### UTILS ##################################################################
#############################################################################

def rotate(px, py, angle, ox=0, oy=0):
    angle = np.radians(angle)
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy
    
def rotate_coordinates(df, angle=29):
    df["pickup_longitude"], df["pickup_latitude"] = \
        rotate(df["pickup_longitude"], df["pickup_latitude"], angle)

    df["dropoff_longitude"], df["dropoff_latitude"] = \
        rotate(df["dropoff_longitude"], df["dropoff_latitude"], angle)
    
def parse_hour(datetime):
    left, right = datetime.split(":", 1)
    left_left, h = left.split(" ", 1)
    return int(h)
    
def arg_closest(point, x, y):
    dist = np.abs(point[:, [0]] - x) + np.abs(point[:, [1]] - y)
    amin = dist.argmin(axis=1)
    return amin, dist[np.arange(len(point)), amin]
    
def save_graph(graph, data_points, name, title, random_colors=False):
    print("saving graph to %s" % name)
    x = data_points[:, 0]
    y = data_points[:, 1]

    lines = []
    dists = []
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j] != REALLY_BIG_NUM and (i < j or graph[i, j] == REALLY_BIG_NUM):
                lines.append(((x[i], y[i]), (x[j], y[j])))
                dists.append(graph[i, j])

    if random_colors:
        colors = np.random.rand(len(lines), 3)
    else:
        dists = np.array(dists)
        min_dists = dists.min()
        colors = np.zeros((len(lines), 3))
        colors[:, 0] = (dists - min_dists)/(dists.max() - min_dists)

    lc = mc.LineCollection(lines, colors=colors)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    fig.suptitle(title)
    fig.savefig(name, dpi=300)

#############################################################################
#### BUILD A GRID THAT REFLECTS NYC DENSITY OF PICKUP/DROPOFF COORDINATES ###
#############################################################################

# First we read a large sample of coordinates
coord_cols = ["pickup_longitude", "pickup_latitude","dropoff_longitude", "dropoff_latitude"]
df = pd.read_csv(inp, nrows=150000, header=0, doublequote=False,
                 quoting=3, usecols=coord_cols)

df = df.dropna()
df = df[df["pickup_longitude"].between(-74.4, -72.9) & \
        df["pickup_latitude"].between(40.5, 41.7) & \
        df["dropoff_longitude"].between(-74.4, -72.9) & \
        df["dropoff_latitude"].between(40.5, 41.7)]

coord1 = df[["pickup_longitude", "pickup_latitude"]].values
coord2 = df[["dropoff_longitude", "dropoff_latitude"]].values
all_coords = np.concatenate([coord1, coord2], axis=0)
all_coords = shuffle(all_coords)
x = all_coords[:, 0]
y = all_coords[:, 1]

x, y = rotate(x, y, angle=29)

# 2d histogram initialization
min_x = x.min(axis=0)
max_x = x.max(axis=0)
min_y = y.min(axis=0)
max_y = y.max(axis=0)
scale_x = max_x - min_x
scale_y = max_y - min_y
scale_diff = (scale_x - scale_y)/2
if scale_diff > 0:
    min_y -= scale_diff
    max_y += scale_diff
else:
    min_x += scale_diff
    max_x -= scale_diff

bins = []
x_edges = []
y_edges = []
for divide in (2, 8, 32, 256):
    hist = np.zeros((divide-1, divide-1))
    boundaries1 = np.linspace(min_x, max_x, num=divide)
    boundaries2 = np.linspace(min_y, max_y, num=divide)
    bins.append(hist)
    x_edges.append(boundaries1)
    y_edges.append(boundaries2)

# bin counting
for b, x_e, y_e in zip(bins, x_edges, y_edges):
    c1 = np.searchsorted(x_e, x) - 1
    c2 = np.searchsorted(y_e, y) - 1
    for n, m in zip(c1, c2):
        b[n, m] += 1

print("counting done")

# grid points are located where the bin count is over a threshold
min_freq = 100
points = []
for b, x_e, y_e in zip(bins, x_edges, y_edges):
    for i, x1 in enumerate(x_e[:-1]):
        x2 = x_e[i+1]
        x12 = (x1 + x2)/2
        for j, y1 in enumerate(y_e[:-1]):
            y2 = y_e[j+1]
            y12 = (y1 + y2)/2
            if b[i, j] > min_freq:
                points.append([x12, y12])

print("number of points", len(points))
data_points = np.array(points)

#############################################################################
#### BUILD A CONNECTED GRAPH FROM THE GRID ##################################
#############################################################################

REALLY_BIG_NUM = 10000000

def add_symmetric_arcs(graph):
    for i in range(len(graph)):
        for j in range(len(graph)):
            graph[i, j] = min(graph[i, j], graph[j, i])

def change_edge_values(graph, from_value, to_value):
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i, j] == from_value:
                graph[i, j] = to_value

def compute_mst(graph):
    dim = len(graph)

    add_symmetric_arcs(graph)

    # prune lots of edges thanks MST algorithm
    change_edge_values(graph, REALLY_BIG_NUM, 0)
    spanning = minimum_spanning_tree(graph, overwrite=True).todense()
    change_edge_values(spanning, 0, REALLY_BIG_NUM)

    add_symmetric_arcs(spanning)

    return spanning
    
dim = len(data_points)
ows = np.arange(dim)

# compute distance between every point
x = data_points[:, 0]
y = data_points[:, 1]

x_diff = x.reshape(dim, 1) - x.reshape(1, dim)
y_diff = y.reshape(dim, 1) - y.reshape(1, dim)

# scaling: move values to a NN-friendly range
coord_scaling = (np.std(x_diff) + np.std(y_diff))/2
x_diff /= coord_scaling
y_diff /= coord_scaling
data_points /= coord_scaling

abs_diff = np.abs(x_diff) + np.abs(y_diff)
dist = np.sqrt(0.0000 + np.square(x_diff) + np.square(y_diff))

# add some noise to dist so that shortest paths will be less sensitive to order
abs_diff += 0.0001 * abs_diff * np.random.normal(size=len(dist))

# compute the angle between every point
angles = np.arctan2(y_diff, x_diff)

# connect points vertically
graph = np.empty((dim, dim))
graph.fill(REALLY_BIG_NUM)

for case, a in ((np.pi/2, angles), (-np.pi/2, angles)):
    criterion = (0.01 + np.abs(a - case)) * dist
    closest = np.argsort(criterion, axis=1)[:, 1:2]
    for i, c in enumerate(closest):
        graph[i, c] = abs_diff[i, c]

mst1 = compute_mst(graph)

# connect points horizontally
graph = np.empty((dim, dim))
graph.fill(REALLY_BIG_NUM)
for case, a in ((0, angles), (np.pi, np.abs(angles))):
    criterion = (0.01 + np.abs(a - case)) * dist
    closest = np.argsort(criterion, axis=1)[:, 1:2]
    for i, c in enumerate(closest):
        graph[i, c] = abs_diff[i, c]

mst2 = compute_mst(graph)

# merge the two MSTs
G = np.empty((dim, dim))
for i in range(dim):
    for j in range(dim):
        G[i, j] = min(mst1[i, j], mst2[i, j])

add_symmetric_arcs(G)

# save an image of the graph
save_graph(G, data_points, "nyc1.png", "graph edges", random_colors=True)

#############################################################################
#### ADJUST DISTANCES WITH DIJKSTRA AND PYTORCH #############################
#############################################################################

# disclaimer: this section requires more work. 
# The cool part is that PyTorch's dynamic graph nicely keeps track of the paths
# between sources and targets without the need for an external memory or an
# additional traversal of the graph. That would also work with other algorithms
# than Dijkstra's, e.g. Floyd's.
# The downside is that the algorithm backpropagates through the putative
# shortest paths only, ignoring other paths. As a result, if an edge gets
# assigned with an overly long distance, it may reach a point of no return
# after which it won't ever be part of any shortest path, and consequently will
# never find its distance decreased by gradient descent. This is why distances
# are clipped in a tight range.
# This is by no means perfect. Perhaps a better approach would be to run
# Dijkstra over a subset of the original graph with randomly sampled edges. In
# this way, every edge could eventually get their chance. 


def torch_dijsktra(gr):
    graph_dim = len(gr)
    graph_copy = [[None] * graph_dim for _ in range(graph_dim)]

    distances = np.empty(graph_dim, dtype=np.float32)
    for i, edges in enumerate(gr):
        distances.fill(REALLY_BIG_NUM)
        distances[i] = 0
        torch_dist = graph_copy[i]
        torch_dist[i] = V(torch.zeros(1))
        for _ in range(graph_dim):
            v = distances.argmin()
            v_dist = torch_dist[v]
            distances[v] = np.inf  # won't be selected by argmin , i.e. removed from the pool
            for neighbor, d, min_d in gr[v]:
                new_d = v_dist + d.clamp(min=0.95, max=1.3) * min_d
                existing_d = torch_dist[neighbor]
                if existing_d is None or (new_d < existing_d).all():
                    torch_dist[neighbor] = new_d
                    distances[neighbor] = new_d.data.numpy()[0]

    return graph_copy
    
def loss(dist_matrix, closest_pickup, closest_dropoff, extra, scaling, y_true, clip=20):
    distances = V(torch.zeros(len(closest_pickup)))
    for n, (cp, cd) in enumerate(zip(closest_pickup, closest_dropoff)):
        distances[n] = dist_matrix[cp][cd]

    y_pred = 2.5 + (distances + extra) * scaling
    error = (y_pred - y_true).abs().clamp(max=clip).mean()
    
    return error

# trips that are likely to be fixed fares will be filtered out
# (e.g. JFK to Manhattan)
# I've found them by couting unique fares with pandas
fixed_fares = np.array([57.33, 49.80, 45.00, 52.00, 49.57, 56.80, 57.54, 49.15])

# select night trips only
# ideally we would build a different map for every hour or every period of time
# that has low traffic variation
night = np.array([0, 1, 2, 3, 4, 5])

# read relevant data
expected = 1000000
gen = pd.read_csv(inp, quoting=3, header=0, doublequote=False, chunksize=2048*32)
size = 0
batches = []
for df in gen:
    print("%i/%i" % (size, expected))
    df = df.dropna()
    hours = list(map(parse_hour, df["pickup_datetime"]))
    fares = df["fare_amount"].values
    df = df[np.isin(hours, night, assume_unique=True) & \
            np.isin(fares, fixed_fares, assume_unique=True, invert=True) & \
            (df["passenger_count"] <= 3 & \
             df["pickup_longitude"].between(-74.4, -72.9) & \
             df["pickup_latitude"].between(40.5, 41.7) & \
             df["dropoff_longitude"].between(-74.4, -72.9) & \
             df["dropoff_latitude"].between(40.5, 41.7) & \
             df["fare_amount"].between(3, 250)).values]

    rotate_coordinates(df)
    
    pickup = df[["pickup_longitude", "pickup_latitude"]].values.astype(np.float32)
    dropoff = df[["dropoff_longitude", "dropoff_latitude"]].values.astype(np.float32)
    targets = df["fare_amount"].values.astype(np.float32)

    batches.append((pickup, dropoff, targets))
    size += len(targets)
    if size >= expected:
        break

pickup, dropoff, targets = zip(*batches)
pickup = np.concatenate(pickup, axis=0)
dropoff = np.concatenate(dropoff, axis=0)
targets = np.concatenate(targets)
print("has read", size, "rows")

pickup /= coord_scaling
dropoff /= coord_scaling

# broadcastable coordinates
x = data_points[:, 0].reshape((1, len(data_points)))
y = data_points[:, 1].reshape((1, len(data_points)))

# for each training data point, find the closest "hub" in the grid
closest_pickup, d1 = arg_closest(pickup, x, y)
closest_dropoff, d2 = arg_closest(dropoff, x, y)
    
# remove data points where dropoff = pickup on the grid
w = np.where(closest_pickup != closest_dropoff)[0]
closest_pickup = closest_pickup[w]
closest_dropoff = closest_dropoff[w]
d1 = d1[w]
d2 = d2[w]
targets = targets[w]
pickup = pickup[w]
dropoff = dropoff[w]
print(len(w), "rows where grid(dropoff) != grid(pickup)")
    
# narrow the optimization to the 5% most accurate cases
# (we don't want to alter distances for trips that don't fit the grid)
ideal = np.abs(pickup - dropoff).sum(axis=1)
di = d1 + d2
hub_dist = np.abs(data_points[closest_pickup] - data_points[closest_dropoff]).sum(axis=1)
ratio = (di+hub_dist)/ideal
print("ratios", ratio)
top = ratio.argsort()[:len(di)//20]
closest_pickup = closest_pickup[top]
closest_dropoff = closest_dropoff[top]
di = di[top].astype(np.float32)
targets = targets[top]

# split set into validation batch and training set
vcut = 16384
v_targets = torch.from_numpy(targets[:vcut])
v_di = torch.from_numpy(di[:vcut])
if gpu >= 0:
    v_targets = v_targets.cuda(gpu)
    v_di = v_di.cuda(gpu)
v_targets = V(v_targets)
v_di = V(v_di)

validation = (closest_pickup[:vcut], closest_dropoff[:vcut], v_di, v_targets)
targets = targets[vcut:]
closest_pickup = closest_pickup[vcut:]
closest_dropoff = closest_dropoff[vcut:]
di = di[vcut:]

# initialize graph with distances as parameters
parameter_graph = []
parameters = []
for i in range(len(G)):
    edges = []
    parameter_graph.append(edges)
    for j in range(len(G)):
        if G[i][j] < REALLY_BIG_NUM:
            d = G[i][j]
            param = nn.Parameter(torch.Tensor([1]))
            if gpu >= 0:
                param = param.cuda(gpu)
            parameters.append(param)
            edges.append((j, param, d))

# parameter to convert distances to fares
# (technically not needed with unconstrained distances
#  but this parameter makes the NN converge faster)
scaling = nn.Parameter(torch.Tensor([10]))
if gpu >= 0:
    scaling = scaling.cuda(gpu)
    
# this first optimizer is to quickly ajust the scaling parameter
optimizer = torch.optim.Adam([scaling], lr=0.05)

# optimization loop
print("start optimizing with", len(targets), "training rows")
max_duration = 3*60*60  # 3 hours
batch_size = 256
switch = 1
for epoch in range(4):
    # stop optimizing if it has already been taking around 3 hours
    if time.time() - start_time > max_duration:
        print("optimization timeout")
        break
    
    if epoch == switch:
        print("switching to optimization with all parameters")
        optimizer = torch.optim.SGD([{"params": [scaling]}, {"params": parameters}], lr=0.15)

    targets, closest_pickup, closest_dropoff, di = \
        shuffle(targets, closest_pickup, closest_dropoff, di)
    
    for i in range(0, len(targets), batch_size):
        if time.time() - start_time > max_duration:
            break
            
        # input batch
        closest_pickup_b = closest_pickup[i:i+batch_size]
        closest_dropoff_b = closest_dropoff[i:i+batch_size]
    
        # convert to tensor
        y_true = torch.from_numpy(targets[i:i+batch_size])
        di_b = torch.from_numpy(di[i:i+batch_size])
        if gpu >= 0:
            y_true = y_true.cuda(gpu)
            di_b = di_b.cuda(gpu)
        y_true = V(y_true)
        di_b = V(di_b)
        
        # forward
        dist_matrix = torch_dijsktra(parameter_graph)
        error = loss(dist_matrix, closest_pickup_b, closest_dropoff_b, di_b, scaling, y_true)
        
        # backward
        optimizer.zero_grad()
        error.backward()
        nn.utils.clip_grad_norm_(parameters, 10)
        optimizer.step()
    
        # reporting with validation batch
        closest_pickup_b, closest_dropoff_b, di_b, y_true = validation
        dist_matrix = torch_dijsktra(parameter_graph)
        error = loss(dist_matrix, closest_pickup_b, closest_dropoff_b, di_b, scaling, y_true, clip=1000000)
        np_loss = float(error.data.cpu().numpy())
        print("loss [epoch %i]" % epoch, np_loss)
        print("scaling", scaling.data.cpu().numpy())
        

print("optimization done")  # expected final loss: around 2.35, which is still too high

# transform parameter graph into drawable graph
G = REALLY_BIG_NUM * np.ones((dim, dim))
for i, edges in enumerate(parameter_graph):
    for j, d, min_d in edges:
        G[i, j] = (d.clamp(min=0.9) * min_d).data.cpu().numpy()

save_graph(G, data_points, "nyc_%f_a.png" % np_loss, "absolute distances [loss: %f]" % np_loss, random_colors=False)

for i, edges in enumerate(parameter_graph):
    for j, d, min_d in edges:
        G[i, j] = d.clamp(min=0.9).data.cpu().numpy()

save_graph(G, data_points, "nyc_%f_b.png" % np_loss, "distance adjustements [loss: %f]" % np_loss, random_colors=False)

### YOU CAN NOW USE NETWORKX TO FIND SHORTEST PATHS