
import pandas as pd
import sqlite3

con = sqlite3.connect('../input/database.sqlite')

query='''
select
    d.anonymous_5, d.anonymous_6, d.anonymous_7,
    c.anonymous_5, c.anonymous_6, c.anonymous_7
from devices d, cookies c
where d.drawbridge_handle != '-1' and d.drawbridge_handle = c.drawbridge_handle;
'''

pos_train_data = pd.read_sql_query(query, con)
print(pos_train_data.head())

query='''
select
    d.anonymous_5, d.anonymous_6, d.anonymous_7,
    c.anonymous_5, c.anonymous_6, c.anonymous_7
from 
    (select * from devices where drawbridge_handle != '-1' and random() % 100 = 0 limit 10) d,
    (select * from cookies where drawbridge_handle != '-1' and random() % 100 = 0 limit 10) c
where d.drawbridge_handle != c.drawbridge_handle;
'''

neg_train_data = pd.read_sql_query(query, con)
print(neg_train_data.head())

'''
d.device_type, d.device_os, d.country, d.anonymous_c0, d.anonymous_c1, d.anonymous_c2, cast(d.anonymous_5 as int), cast(d.anonymous_6 as int), cast(d.anonymous_7 as int),
c.computer_os_type, c.browser_version, c.anonymous_c0, c.anonymous_c1, c.anonymous_c2, cast(c.anonymous_5 as int), cast(c.anonymous_6 as int), cast(c.anonymous_7 as int)
'''