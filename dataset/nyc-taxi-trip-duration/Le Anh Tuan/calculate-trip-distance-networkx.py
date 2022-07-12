# Multi-process version
# Calculate drive distance using OpenStreetMap and NetworkX
# data processing
# Credit to:
# https://www.kaggle.com/ankasor/driving-distance-using-open-street-maps-data/notebook
# https://pymotw.com/2/multiprocessing/communication.html
# change DATA_DIR to folder containing trainging data set
import pandas as pd
import numpy as np

# System
import datetime as dtime
from datetime import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
import logging
import multiprocessing

# Other
from geographiclib.geodesic import Geodesic
import osmnx as ox
import networkx as nx

DATA_DIR = "../input"
STREETGRAPH_FILENAME = 'streetnetwork.graphml'
LOCATION = 'New York, USA'
LOG_LEVEL = logging.DEBUG
# LOG_LEVEL = logging.INFO
NUM_THREAD = 4
BLOCK_SIZE = 100


class DriveDistance(multiprocessing.Process):
    def __init__(self, thread_count=0, in_queue=None, out_queue=None, out_queue2=None, manager_dict=None, mode=0):
        multiprocessing.Process.__init__(self)
        self.thread_count = thread_count
        self.mode = mode
        self.graph_filename = DATA_DIR + "/" + STREETGRAPH_FILENAME
        self.train_distance_filename = DATA_DIR + "/train_distance.csv"
        self.eval_distance_filename = DATA_DIR + "/test_distance.csv"
        self.distance_filename = DATA_DIR + "/distance.csv"
        self.geod = Geodesic.WGS84  # define the WGS84 ellipsoid
        if in_queue is not None:
            self.in_queue = in_queue
        if out_queue is not None:
            self.out_queue = out_queue
        if out_queue2 is not None:
            self.out_queue2 = out_queue2
        if manager_dict is not None:
            self.manager_dict = manager_dict
        # logger.info("Process started " + str(thread_count))

    def run(self):
        self.logger = self.get_logger()
        if self.mode == 0:
            self.run_main()
        else:
            self.run_worker()

    def run_main(self):
        # Load graph and train+eval set => put to shared dictionary
        area_graph = self.init_osm_graph()
        combine_data = self.load_data()
        manager_dict['area_graph'] = area_graph
        manager_dict['combine_data'] = combine_data
        self.out_queue2.put(1)
        self.save_distance()

    def run_worker(self):
        while True:
            # Get the work from the queue and expand the tuple
            data = self.in_queue.get()
            if data is None:
                self.logger.info(str(self.thread_count) +
                                 " No task remain. Existing ...")
                self.in_queue.task_done()
                break
            self.logger.info(str(self.thread_count) +
                             " got data:" + str(len(data)))
            self.calc_drive_distance(data)
            self.in_queue.task_done()

    def get_logger(self):
        logger = logging.getLogger('kaggle-' + str(self.thread_count))
        # logger = multiprocessing.get_logger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(
            DATA_DIR + '/drive_distance_' + str(self.thread_count) + '.log', mode='a')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        return logger

    def get_area_graph(self):
        return self.manager_dict['area_graph']

    def get_combine_data(self):
        return self.combine_data

    def init_osm_graph(self):
        # self.logger = self.get_logger()
        if not os.path.isfile(self.graph_filename):
            # There are many different ways to create the Network Graph. See
            # the osmnx documentation for details
            self.logger.info(str(self.thread_count) +
                             " Downloading graph for " + LOCATION)
            area_graph = ox.graph_from_place(
                LOCATION, network_type='drive_service')
            ox.save_graphml(
                area_graph, filename=STREETGRAPH_FILENAME, folder=DATA_DIR)
            self.logger.info(str(self.thread_count) +
                             " Graph saved to " + self.graph_filename)
        else:
            self.logger.info(str(self.thread_count) +
                             " Loading graph from " + self.graph_filename)
            area_graph = ox.load_graphml(
                STREETGRAPH_FILENAME, folder=DATA_DIR)
        return area_graph

    def load_data(self):
        # self.logger = self.get_logger()
        self.logger.info(str(self.thread_count) +
                         " Loading data from " + DATA_DIR)
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        eval_data = pd.read_csv(DATA_DIR + "/test.csv")
        features = eval_data.columns.values
        train_data = train_data[features]
        combine_data = pd.concat(
            [train_data[features], eval_data])
        return combine_data
        # self.combine_data = self.combine_data[:11]

    def point_distance(self, startpoint, endpoint):
        distance = self.geod.Inverse(
            startpoint[0], startpoint[1], endpoint[0], endpoint[1])
        return distance['s12']

    def driving_distance(self, area_graph, startpoint, endpoint):
        """
        Calculates the driving distance along an osmnx street network between two coordinate-points.
        The Driving distance is calculated from the closest nodes to the coordinate points.
        This can lead to problems if the coordinates fall outside the area encompassed by the network.

        Arguments:
        area_graph -- An osmnx street network
        startpoint -- The Starting point as coordinate Tuple
        endpoint -- The Ending point as coordinate Tuple
        """
        # Find nodes closest to the specified Coordinates
        node_start = ox.utils.get_nearest_node(area_graph, startpoint)
        node_stop = ox.utils.get_nearest_node(area_graph, endpoint)
        # Calculate the shortest network distance between the nodes via the edges
        # "length" attribute
        try:
            distance = nx.shortest_path_length(
                area_graph, node_start, node_stop, weight="length")
        except:
            self.logger.error(str(self.thread_count) + " Can not calculate path from (" + str(startpoint[0]) +
                              "," + str(startpoint[0]) + ")" + " to (" +
                              str(endpoint[0]) + "," +
                              str(endpoint[1]) + "). Using fallback function")
            distance = self.point_distance(startpoint, endpoint)
        return distance

    def calc_drive_distance(self, data):
        # data_index = data.index.values
        # print(self.thread_count, len(data), data_index)
        start = time.time()
        area_graph = self.get_area_graph()
        distance = data.apply(lambda row: self.driving_distance(
            area_graph, (row['pickup_latitude'],
                         row['pickup_longitude']),
            (row['dropoff_latitude'], row['dropoff_longitude'])),
            axis=1)
        # print("data size:", len(data), " distance size:", len(distance))
        # data.loc[n_start:n_end, 'drive_distance'] = distance.values
        # self.combine_data.loc[n_start:n_end, 'drive_distance'] = distance.values
        column = 'drive_distance'
        distance_pd = pd.DataFrame(
            data={'id': data['id'], column: distance.values}, columns=['id', column])
        self.out_queue.put(distance_pd)
        end = time.time() - start
        self.logger.info(str(self.thread_count) +
                         " processed time:" + str(end))

    def save_distance(self):
        self.logger.info(str(self.thread_count) + " Saving to " + DATA_DIR)
        column = 'drive_distance'
        distance = pd.DataFrame(columns=['id', column])
        while True:
            data = self.out_queue.get()
            if data is not None:
                distance = distance.append(data)
                distance.to_csv(
                    self.distance_filename, index=False)
                self.logger.info(str(self.thread_count) +
                                 " total saved: " + str(len(distance)))
            else:
                self.logger.info(str(self.thread_count) +
                                 " Total data:" + str(len(distance)))
                break



# ---------------- Main -------------------------
if __name__ == "__main__":
    multiprocessing.log_to_stderr(logging.DEBUG)
    # create logger
    logger = logging.getLogger('kaggle2')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DATA_DIR + '/drive_distance.log', mode='a')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Create global queue
    in_queue = multiprocessing.JoinableQueue()
    out_queue = multiprocessing.Queue()
    out_queue2 = multiprocessing.Queue()

    # Create manager
    manager = multiprocessing.Manager()
    manager_dict = manager.dict()
    # create base class
    base_class = DriveDistance(
        0, in_queue=in_queue, out_queue=out_queue, out_queue2=out_queue2, manager_dict=manager_dict, mode=0)
    # area_graph = base_class.init_osm_graph()
    # combine_data = base_class.load_data()
    # manager_dict['area_graph'] = area_graph
    base_class.start()
    # Wait until base_class done loading data and map
    result = out_queue2.get()
    # create worker
    workers = [DriveDistance(x + 1, in_queue=in_queue, out_queue=out_queue, out_queue2=out_queue2, manager_dict=manager_dict, mode=1)
               for x in range(NUM_THREAD)]
    logger.info("Total workers:" + str(len(workers)))
    for w in workers:
        w.start()
    ts = time.time()
    combine_data = manager_dict['combine_data']
    data_len = len(combine_data)
    # data_len = 100
    remain = data_len % BLOCK_SIZE
    total_blocks = data_len // BLOCK_SIZE
    if remain > 0:
        total_blocks = total_blocks + 1
    logger.info("Data size: " + str(data_len) +
                " total blocks:" + str(total_blocks))
    for n_block in range(total_blocks):
        n_start = n_block * BLOCK_SIZE
        n_end = n_start + BLOCK_SIZE
        data = combine_data[n_start:n_end]
        in_queue.put(data)

    # Add a poison pill for each consumer
    for i in range(NUM_THREAD):
        in_queue.put(None)
    # Causes the main thread to wait for the queue to finish processing all
    # the tasks
    in_queue.join()
    logger.info('Took ' + str(time.time() - ts))
    # Signal base_class to terminate
    out_queue.put(None)
    # Wait for all process terminated
    time.sleep(2)
    logger.info('Existing ...')
    # base_class.save_distance()
