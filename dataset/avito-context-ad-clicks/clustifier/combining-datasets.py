# -*- coding: utf-8 -*-
import sqlite3
import csv, sys


TITLE = ['SearchID', 'AdID', 'Position', 'ObjectType', 'HistCTR', 'IsClick',
         'SearchDate', 'IPID', 'UserID', 'IsUserLoggedOn', 'SearchQuery',
         'SearchLocationID', 'SearchCategoryID', 'SearchParams', 'UserAgentID',
         'UserAgentFamilyID', 'UserAgentOSID', 'UserDeviceID', 'Params',
         'Price', 'Title', 'IsContext', 'CLevel', 'ParentCategoryID',
         'SubcategoryID', 'LLevel', 'RegionID', 'CityID']

conn = sqlite3.connect("../input/database.sqlite")
SearchStream = conn.cursor()
SearchInfo = conn.cursor()
UserInfo = conn.cursor()
AdsInfo = conn.cursor()
Category = conn.cursor()
Location = conn.cursor()
for _ in SearchStream.execute("select * from trainSearchStream tss left join SearchInfo si "+
    "on tss.SearchID=si.SearchID where ObjectType==3 and UserID=1504499 "):
    print (_)
sys.exit(0)
for _ in SearchStream.execute("select count(*) from trainSearchStream where ObjectType==3"):
    print (_)
sys.exit(0)
output_file = open("train.csv", "w", encoding="utf8")
open_file_object = csv.writer(output_file)
open_file_object.writerow(TITLE)

search = SearchStream.fetchmany(10000)
cnt = len(search)
rows = []
while search:
    for i in search:
        print(i)
        if i[5] == '':
            continue
        search_id = i[1]
        ad_id = i[2]
        SearchInfo.execute("select * from SearchInfo where SearchID="+str(search_id))
        AdsInfo.execute("select * from AdsInfo where AdID="+str(ad_id))
        search_info = SearchInfo.fetchone()
        ads_info = AdsInfo.fetchone()
        if search_info is None:
            if ads_info is None:
                ads_info = [0 for k in range(7)]
            search_info = [0 for k in range(9)]
            user_info = [0 for k in range(5)]
            category = [0 for k in range(4)]
            location = [0 for k in range(4)]
            row = list(i) + list(search_info[1:]) + list(user_info[1:]) + \
                  list(ads_info[3:]) + list(category[1:]) + list(location[1:])
            rows.append(row)
            continue
        user_id = search_info[3]
        location_id = search_info[6]
        category_id = search_info[7]
        try:
            UserInfo.execute("select * from UserInfo where UserID="+str(user_id))
            Category.execute("select * from Category where CategoryID="+str(category_id))
            Location.execute("select * from Location where LocationID="+str(location_id))
        except:
            UserInfo.execute("select * from UserInfo where UserID="+str(user_id))
            Category.execute("select * from Category where CategoryID="+str(category_id))
            Location.execute("select * from Location where LocationID="+str(location_id))
        user_info = UserInfo.fetchone()
        category = Category.fetchone()
        location = Location.fetchone()
        if ads_info is None:
            ads_info = [0 for k in range(7)]
        if user_info is None:
            user_info = [0 for k in range(5)]
        if category is None:
            category = [0 for k in range(4)]
        if location is None:
            location = [0 for k in range(4)]
        row = list(i) + list(search_info[1:]) + list(user_info[1:]) + \
              list(ads_info[3:]) + list(category[1:]) + list(location[1:])
        rows.append(row)
    print(cnt)
    open_file_object.writerows(rows)
    rows = []
    search = SearchStream.fetchmany(10000)
    cnt += len(search)

output_file.close()
