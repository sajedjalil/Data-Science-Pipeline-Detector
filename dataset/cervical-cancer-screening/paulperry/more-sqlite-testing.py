# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) 
# will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import datetime
start = datetime.datetime.now()
print(start)

import sqlite3
db = sqlite3.connect('../input/database.sqlite')

# get list of tables
# table = pd.read_sql_query("SELECT * FROM sqlite_master;", db)
# print(table)

# get schema 
# for i in table.index:
#     print(table.sql[i])

# print schema
#
# def printSchema(connection):
#    for (tableName,) in connection.execute(
#        """
#        select NAME from SQLITE_MASTER where TYPE='table' order by NAME;
#        """
#    ):
#        print("{}:".format(tableName))
#        for (
#            columnID, columnName, columnType,
#            columnNotNull, columnDefault, columnPK,
#        ) in connection.execute("pragma table_info('{}');".format(tableName)):
#            print("  {id}: {name}({type}){null}{default}{pk}".format(
#                id=columnID,
#                name=columnName,
#                type=columnType,
#                null=" not null" if columnNotNull else "",
#                default=" [{}]".format(columnDefault) if columnDefault else "",
#                pk=" *{}".format(columnPK) if columnPK else "",
#            ))
#
# printSchema(db)

# for row in db.execute("pragma table_info('sqlite_master')").fetchall():
#     print(row)

train = pd.read_sql_query('select * from patients_train limit 10;', db)
print(train)

### END
