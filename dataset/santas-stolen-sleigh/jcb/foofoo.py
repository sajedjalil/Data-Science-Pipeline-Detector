import sqlite3
import pandas as pd
from haversine import haversine
from sys import exit


north_pole = (90,0)
weight_limit = 1000.0
print(haversine((0,0), north_pole))

exit(0)
