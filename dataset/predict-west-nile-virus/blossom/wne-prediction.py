# WNE prediction challenge
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.
# Descriptive study in west nile virus prediction

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KernelDensity
from subprocess import check_output


def plot_heat_map(data, a_cm):
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    if(len(data) > 0):
        X = data[['Longitude', 'Latitude']].values
        kd = KernelDensity(bandwidth=0.02)
        kd.fit(X)
        xv,yv = np.meshgrid(np.linspace(-88, -87.5, 100), np.linspace(41.6, 42.1, 100))
        gridpoints = np.array([xv.ravel(),yv.ravel()]).T
        zv = np.exp(kd.score_samples(gridpoints).reshape(100,100))
        plt.imshow(zv, 
                   origin='lower', 
                   cmap=a_cm, 
                   extent=lon_lat_box, 
                   aspect=aspect)
                   
def plot_spray_vs_trap(spray, traps):
    
    alpha_cm = plt.cm.Reds
    alpha_cm._init()
    alpha_cm._lut[:-3,-1] = abs(np.logspace(0, 1, alpha_cm.N) / 10 - 1)[::-1]
    alpha_mcm = plt.cm.Greens
    alpha_mcm._init()
    alpha_mcm._lut[:-3,-1] = abs(np.logspace(0, 1, alpha_cm.N) / 10 - 1)[::-1]
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    subplot = 0
    
    species     = traps['Species'].unique()
    years       = traps['year'].unique()
    num_species = len(species)
    
    plt.figure(figsize=(18,6*num_species))
    for spcsIndx in range(num_species):
        for year in years:
            subplot += 1
            sightings = spray[(spray['year'] == year)]
            mSightings = traps[(traps['Species'] == species[spcsIndx])
                              & (traps['year'] == year)]
            mSightings = mSightings.groupby(['year','month', 'day', 'Trap', 'Longitude', 'Latitude', 'Species']).max()['WnvPresent'].reset_index()
            if(len(mSightings) <= 0):
                print("SKIPPING [" + str(subplot) + "]:" + str(year) + " (" + species[spcsIndx] + ")\t\tNo sightings")
                continue
    
            plt.subplot(num_species, 4, subplot)
            plt.gca().set_title(str(year) + " (" + species[spcsIndx] + ")")
            plt.imshow(mapdata, 
                       cmap=plt.get_cmap('gray'), 
                       extent=lon_lat_box, 
                       aspect=aspect)
            plot_heat_map(mSightings, alpha_cm)
            plot_heat_map(sightings, alpha_mcm)
    
            print("         [" + str(subplot) + "]:" + str(year) + " (" + species[spcsIndx] + ")")
            plt.tight_layout()
            locations = traps[['Longitude', 'Latitude']].drop_duplicates().values
            plt.scatter(locations[:,0], locations[:,1], marker='x')
    
    plt.savefig('heatmap_spray_species_vs_trap.png')
    
def process_date(data):
    '''
    Extract the year, month and day from the date
    '''
    # Functions to extract year, month and day from dataset
    def get_year(x):
        return int(x.split('-')[0])
    def get_month(x):
        return int(x.split('-')[1])
    def get_day(x):
        return int(x.split('-')[2])
    
    data['year' ] = data.Date.apply(get_year)
    data['month'] = data.Date.apply(get_month)
    data['day'  ] = data.Date.apply(get_day)
        
    return data

def read_input_files(dir_path):
    '''
    Read the input files: train.csv, test.csv, sampleSubmission.csv
    '''
    train    = pd.read_csv('%s/train.csv' % dir_path)
    test     = pd.read_csv('%s/test.csv' % dir_path)
    #sample = pd.read_csv('%s/sampleSubmission.csv' % dir_path)
    spray    = pd.read_csv('%s/spray.csv' % dir_path)
    weather  = pd.read_csv('%s/weather.csv' % dir_path)
    mapdata  = np.loadtxt('%s/mapdata_copyright_openstreetmap_contributors.txt' %dir_path)
    return train, test, spray, weather, mapdata
    


if __name__ == "__main__":
    import sys

    # Check command-line arguments
    if len(sys.argv)>1:
        print ('Usage: "python West_nile_virus_prediction.py"')
        exit()
        
    input_dir = '../input/'
    #make_dirs(['Descriptive'])
    print(check_output(["ls", input_dir]).decode("utf8"))
    
    # Load input datasets
    print ('Reading input files')
    train, test, spray, weather,  mapdata = read_input_files(input_dir) 
    print ('There are %d, %d training,test samples' % (len(train), len(test)))

    # Process date
    train = process_date(train)
    test  = process_date(test)
    spray = process_date(spray)
    weather = process_date(weather)
    print ('labels train',train.columns)
    print ('labels test',test.columns)
    print ('labels spray',spray.columns)
    print ('labels weather',weather.columns)
    traps = train[['year','month', 'day', 'Trap','Longitude', 'Latitude', 'Species', 'WnvPresent']]
    # Heatmap at year in train
    plot_spray_vs_trap(spray, traps)
    
    print (len(traps['Trap'].unique()))
    print (len(test['Trap'].unique()))
    
    print ('years weather',weather['year'].unique())
    print ('years spray',spray['year'].unique())
    print ('heat data:', weather['Heat'].unique())
    print ('codesum data:', weather['CodeSum'].unique())
    print ('StnPressure:', weather['StnPressure'].unique())
    print ('SeaLevel:', weather['SeaLevel'].unique())
    print ('AvgSpeed:', weather['AvgSpeed'].unique())
    
