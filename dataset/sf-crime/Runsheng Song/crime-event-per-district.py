import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import zipfile
style.use('ggplot')

def densityOfDistrict():
    '''
    crime event of each district
    '''
    
    #group crime incident by districts
    districts = np.unique(train['PdDistrict'])
    dist_freq = train[['X','PdDistrict']].groupby(['PdDistrict']).count().rename(columns={'X':'CrimeApp'})
    plt.figure()
    dist_freq.plot(kind='bar')
    plt.show()
    plt.savefig('Districts.png')
    
    
def appOfCate():
    '''
    crime evet per category
    '''
    # train = pd.read_csv('./data/train.csv',parse_dates=['Dates'])
    #group crime incident by districts
    CrimeFreq = train.groupby(['Category'])
    CrimeFreq = CrimeFreq['X'].count()
    plt.figure()
    CrimeFreq.plot(kind='bar',color='blue')
    plt.show()
    plt.savefig('CrimeAppearance.png')
    
def crimeAppPerDist():
    '''
    crime event per distrit per category
    '''
    
    train['event'] = 1
    CrimeApp = train[['PdDistrict','Category']]
    distName = np.unique(train['PdDistrict'])
    dist_crime_event = train[['PdDistrict','Category','event']].groupby(['PdDistrict','Category']).count().reset_index()
    thePivot = dist_crime_event.pivot(index='Category', columns='PdDistrict',values='event').fillna(0)
    plt.figure()
    thePivot.plot(kind='bar')
    plt.show()
    plt.savefig('Crime_District.png')
    
if __name__ == '__main__':
    z = zipfile.ZipFile('../input/train.csv.zip')
    train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])
    densityOfDistrict()
    appOfCate()
    crimeAppPerDist()
    
    '''interesting finding'''
    '''
    Southern district has the highest event of crime
    However tenderlion stands out in DRUG/NARCOTIC crime
    '''
    
    