import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite:///../input/database.sqlite')

def get_count(name):
    count=pd.read_sql('select count(*) from '+name, engine)
    return count.iloc[0,0]
def get_sample(name):
    sample=pd.read_sql('select * from ' + name + ' limit 10', engine)
    return sample 
tabs=["AdsInfo","Category","Location","PhoneRequestsStream","SearchInfo","UserInfo","VisitsStream","testSearchStream","trainSearchStream"]
counts=[get_count(tab) for tab in tabs]

tab_count_df=pd.DataFrame({'name': tabs, 'row_count':counts})
tab_count_df

samples=[get_sample(tab) for tab in tabs]
sample_df=pd.DataFrame({'name': tabs, 'sample':samples})
sample_df.iloc[0,1] #AdsInfo