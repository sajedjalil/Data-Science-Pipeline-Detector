import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from mpl_toolkits.basemap import Basemap

def inputfile():
    file = zipfile.ZipFile('../input/train.csv.zip')
    data = pd.read_csv(file.open('train.csv'))
    return data

def categoryCrime(data):
    #print(data)
    categories = {}
    print(data.columns)
    category = data['Category']
    for cate in category:
        if cate in categories:
            categories[cate] += 1
        else:
            categories[cate] = 1
    print(categories)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 15.5)

    plt.bar(range(1,len(categories)*2,2), categories.values(), align='center')
    plt.xticks(range(1,len(categories)*2,2), list(categories.keys()), rotation = 'vertical')
    #plt.show()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig('crime.png')
    #categories = data.set_index('Category').T.to_dict('listss')
    #print(listss)
    
def heatSF(data):
    lists = data[['X','Y']].values.tolist()
    pairs = []
    for i in range(len(lists)):
        pairs.append((lists[i][0],lists[i][1]))
    print(pairs)
    #mapSF = gmaps.heatmap(pairs)
    #gmaps.display(mapSF)
    #plt.show()
    
if __name__ == "__main__":
    data = inputfile()
    #categoryCrime(data)
    heatSF(data)
