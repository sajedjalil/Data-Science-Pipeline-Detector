import pandas as pd

#UserInfo.tsv
user_info=pd.read_csv('../input/UserInfo.tsv',delimiter='\t',encoding='utf-8')
print(list(user_info.columns.values)) #file header
print(user_info.tail(35)) #last N rows

#Location.tsv
location=pd.read_csv('../input/Location.tsv',delimiter='\t',encoding='utf-8')
print(list(location.columns.values)) #file header
print(location.tail(35)) #last N rows

#category.tsv
category=pd.read_csv('../input/Category.tsv',delimiter='\t',encoding='utf-8')
print(list(category.columns.values)) #file header
print(category.tail(5)) #last N rows

#big csv file...
search_info = pd.read_csv('../input/SearchInfo.tsv',delimiter='\t',nrows=20,encoding='utf-8-sig')#skiprows=1000000, 
print(list(search_info.columns.values)) #file header
print(search_info[[search_info.columns[0],search_info.columns[1],search_info.columns[2]]])
