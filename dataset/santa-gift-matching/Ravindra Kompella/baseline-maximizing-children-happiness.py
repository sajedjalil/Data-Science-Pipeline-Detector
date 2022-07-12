# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

wList = pd.read_csv('../input/child_wishlist_v2.csv', header = None)
gPref = pd.read_csv('../input/gift_goodkids_v2.csv', header = None)
sample_sub = pd.read_csv('../input/sample_submission_random_v2.csv')

# Set the first column as index
wList.set_index(0, inplace=True)

# Initialize the gift id to -1 for all the children
sample_sub.iloc[:,1] = -1

# Create a gift id array and populate it with 1000. The index of this array gives the gift id
gid_array = [] 
for i in range(1000):
    gid_array.append(1000)
    

def assignGifts_initial():
    # Start first by assigning the gifts to normal kids, then to twins and then to triplets. This maximizes
    # the child happiness
    for i in range(45001,1000000,1):        
        for j in range(100):
            gid = wList.iloc[i,j]
            if (gid_array[gid] > 0):
                gid_array[gid] -= 1
                sample_sub.iloc[i,1] = gid
                break

    # Assign next to twins            
    for i in range(5001,44999,2):
        for j in range(100):
            gid = wList.iloc[i,j]
            if (gid_array[gid] >= 2):
                gid_array[gid] -= 2
                sample_sub.iloc[i,1] = gid
                sample_sub.iloc[i+1,1] = gid
                break

    # Assign next to triplets            
    for i in range(0,4998,3):
        for j in range(100):
            gid = wList.iloc[i,j]
            if (gid_array[gid] >= 3):
                gid_array[gid] -= 3
                sample_sub.iloc[i,1] = gid
                sample_sub.iloc[i+1,1] = gid
                sample_sub.iloc[i+2,1] = gid
                break

def unassignedgiftDict():
    # Populate a dictionary of all the gifts that have remained after the initial assignment
    # This is done for increasing the speed up
    rem_dict = {}
    for k,v in enumerate(gid_array):
        if v > 0:
            rem_dict[k] = v
    return rem_dict


def assignGifts_final():
    # Final round of assignment to the left over kids
    def getGiftIdTriplets():
        # assign gifts to triplets randomly from the remaining gifts
        for k,v in rem_dict.items():
            if(v >= 3):
                giftid = k
                rem_dict[k] -= 3
                break
        return giftid


    def getGiftIdTwins():
        # assign gifts to twins randomly from the remaining gifts
        for k,v in rem_dict.items():
            if(v >= 2):
                giftid = k
                rem_dict[k] -= 2
                break
        return giftid


    def getGiftId():
        # assign gifts to remaining leftover randomly from the remaining gifts
        for k,v in rem_dict.items():
            if(v >= 1):
                giftid = k
                rem_dict[k] -= 1
                break
        return giftid

    # Assign to the triplets who didnt receive gift so far
    rem_triplets = [i for i in sample_sub[sample_sub.GiftId == -1].index if (i >= 0 and i < 5001)]
    for i in [e for e in rem_triplets[::3]]:
        giftid = getGiftIdTriplets()
        sample_sub.iloc[i,1] = giftid
        sample_sub.iloc[i+1,1] = giftid
        sample_sub.iloc[i+2,1] = giftid

    # Assign to the twins who didnt receive gift so far
    rem_twins = [i for i in sample_sub[sample_sub.GiftId == -1].index if (i >= 5001 and i < 45000)]
    for i in [e for e in rem_twins[::2]]:
        giftid = getGiftIdTwins()
        sample_sub.iloc[i,1] = giftid
        sample_sub.iloc[i+1,1] = giftid        
    
    # Assign to the remaining kids who didnt receive gift so far
    rem_kids = [i for i in sample_sub[sample_sub.GiftId == -1].index if i > 45000]
    for i in rem_kids:
        giftid = getGiftId()
        sample_sub.iloc[i,1] = giftid


assignGifts_initial()
print('Done initial assignment')

rem_dict = unassignedgiftDict()
print('Done creating the dictionary')

assignGifts_final()
print('Done final assingment')

print('Total number of children still left over after final assignment', sample_sub[sample_sub.GiftId == -1].count())

sample_sub.to_csv('submit.csv', index = False)



























































            
            