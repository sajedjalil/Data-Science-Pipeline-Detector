import sqlite3
from pandas.io import sql

# Create connection.
cnx = sqlite3.connect('../input/database.sqlite')

#look up table names
#all_tables = sql.read_sql("SELECT * FROM sqlite_master;", cnx)
#print (all_tables)

#query a single table with limits on nrows and ncols
a_table = sql.read_sql("SELECT * FROM UserInfo LIMIT 100 OFFSET 3;", cnx) #LIMIT[nrows] OFFSET [ncols]
print (a_table)

#write data to hdf5, eventually split by looping on nrows
#data.to_hdf('PATH_TO_LOCAL_FOLDER/FILENAME.hdf','test',mode='w')


