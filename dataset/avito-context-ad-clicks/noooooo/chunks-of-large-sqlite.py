import sqlite3
from pandas.io import sql

# Create connection.
cnx = sqlite3.connect('../input/database.sqlite')

#look up table names
all_tables = sql.read_sql("SELECT * FROM sqlite_master;", cnx)
print (all_tables)



