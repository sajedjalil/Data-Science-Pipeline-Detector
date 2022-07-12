
import pandas as pd
import sqlite3

con = sqlite3.connect('../input/database.sqlite')

devices = pd.read_sql_query("select * from devices limit 100;", con)
print(devices.head())
