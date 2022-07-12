import os
import numpy as np
import pandas as pd
import time
from sympy import isprime, primerange
import random
from itertools import permutations
import threading


# Config here. These are low to run fast in a script...
idx_range = 7 # permutations is the factorial of this...
num_threads = 8
offset = 0
max_tries = 100

initial_path = pd.read_csv('../input/brute-force-santa-2018-initial/1515656.6225610063santa2018.csv')
cities = pd.read_csv("../input/traveling-santa-2018-prime-paths/cities.csv")

pnums = [i for i in primerange(0, 197770)]

def score_path(path_df):
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()


path_df = cities.reindex(initial_path['Path']).reset_index()    
start_idx = path_df.loc[path_df['CityId'] == 0].index
a = path_df.iloc[start_idx].iloc[0]
b = path_df.iloc[start_idx].iloc[1]

path_df_best = path_df.copy()
best_score = score_path(path_df_best)
print("Try for a good shuffle")
for i in range(10):
    path_df= path_df_best.copy()
    #shufle
    path_df = path_df.sample(frac=1).reset_index(drop=True)
    # always start at 0
    curr_idx = path_df.loc[path_df['CityId'] == 0].index
    #copy the wrong ones into place
    path_df.iloc[curr_idx[0]] = path_df.iloc[start_idx[0]]
    path_df.iloc[curr_idx[1]] = path_df.iloc[start_idx[1]]
    #start and end to 0
    path_df.iloc[start_idx[0]] = a
    path_df.iloc[start_idx[1]] = b 

    score = score_path(path_df)
    print(str(score).ljust(20) + " / " + str(best_score).ljust(20))
    if score < best_score:
        best_score = score
        path_df_best = path_df.copy()

path_df = path_df_best.copy()
curr_idx = path_df.loc[path_df['CityId'] == 0].index
print(curr_idx)

path_df_best = path_df.copy()
best_score = score_path(path_df_best)
score = best_score
max_idx=path_df['Path'].shape[0]

print( "Initial best: " + str(best_score))
try:
    for ix in range(offset,min(max_tries,int(max_idx-(idx_range+1)))):
        t = time.time()
        path_df = path_df_best.copy() #start at best
        start_loc = random.randint(1,max_idx-(idx_range+1)-1)
        i = start_loc

        rand_idx = random.randint(1,max_idx-(idx_range+1)-1)
        # start indexes are for current best
        start_idxes = list(range(start_loc,start_loc+idx_range))  
        dest_idxes = list(range(start_loc,start_loc+idx_range))  

        #add a random in to get lucky?
        start_idxes.pop()
        dest_idxes.pop()
        start_idxes.append(rand_idx)
        dest_idxes.append(rand_idx)
        # generate all permutations
        pms = np.array(list(set(permutations(dest_idxes))))

        # setup threads
        tmp_scores = num_threads*[None]
        tmp_swaps = num_threads*[None]
        threads = []
        thread_input = num_threads*[None]
        thread_idx = 0
        batch = pms.shape[0] // (num_threads)

        # split permutations into batches per thread
        for start in range(0, pms.shape[0], batch):
            end = min(pms.shape[0]-1, start + batch)
            thread_input[thread_idx]  = pms[start:end, start:end]
            thread_idx += 1

        # worker iterates through permutations and gets the score
        def worker(data_idx):
            wpms = thread_input[data_idx].copy()
            wpath_df = path_df_best.copy()
            scores = []
            swaps=[]
            if wpms is not None:
                for dest_idx in wpms:
                    # skip the known solution
                    if list(start_idxes) == list(dest_idx):
                        continue
                    # swap locations    
                    for i in range(len(dest_idx)):
                        wpath_df.iloc[dest_idx[i]] = path_df_best.iloc[start_idxes[i]]

                    # calculate the score
                    s = score_path(wpath_df)
                    if len(list(dest_idx)) == len(list(start_idxes)): # why would this happen?
                        scores.append(s)
                        swaps.append([list(start_idxes),list(dest_idx)])
            
            # collect the results
            tmp_scores[data_idx] = scores
            tmp_swaps[data_idx] = swaps

        #start threads
        for j in range(num_threads):
            th = threading.Thread(target=worker, args=(j,), daemon=True)
            th.start()
            threads.append(th)

        #join threads, wait for completion
        for th in threads:
            if th is not None:
                th.join()

        # put together results
        x = np.concatenate(tmp_scores)

        # remove empties from the end
        l = tmp_swaps.pop()
        while len(l) == 0:
            l = tmp_swaps.pop()
        tmp_swaps.append(l) #append back the non empty

        s = np.concatenate(tmp_swaps) 
        lowest = np.min(x)
        # print results
        print(str(i).ljust(6) + " : " +str(ix).rjust(6)+" / " +  str(min([int(max_idx-(idx_range+1)),max_tries])).ljust(6) +
            " CITY: " + str(path_df.iloc[i]['Path']).ljust(10) +
            " BATCH_SIZE: "+str(len(pms)))
        print("BEST: " + str(lowest).ljust(19) + " / " + str(best_score).ljust(19) )
        print("PM:" + str(s[np.argmin(x)][1]) )
        print("T:" + str(time.time() - t))
        # save the best if there is a better one
        if lowest < best_score:
            print("NEW BEST: " + str(np.min(x)))
            start_idx = s[np.argmin(x)][0]
            dest_idx = s[np.argmin(x)][1]
            print(start_idx)
            print(dest_idx)
            path_df = path_df_best.copy()
            for i in range(len(dest_idx)):
                path_df.iloc[dest_idx[i]] = path_df_best.iloc[start_idx[i]]
            print(score_path(path_df))
            path_df_best = path_df.copy()
            best_score = lowest
except Exception as e:
    raise e
except: # keyboard exception, save the best result
    out_df = path_df_best[['Path']].astype(int)
    out_df.to_csv(str(best_score)+'santa2018.csv', index=False)
    print("\nSAVED: "+str(best_score)+'santa2018.csv')

# get to the end? save the best
out_df = path_df_best[['Path']].astype(int)
out_df.to_csv(str(best_score)+'santa2018.csv', index=False)
print("\nSAVED: "+str(best_score)+'santa2018.csv')
