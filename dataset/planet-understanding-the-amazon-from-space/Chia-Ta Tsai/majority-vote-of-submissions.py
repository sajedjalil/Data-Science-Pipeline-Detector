import time
from math import ceil
import pandas as pd

target_id = 'image_name'


if __name__ == '__main__':
        
    #read in predictions
    file = 'SubmVoters.csv'
    collection = pd.read_csv(file, skipinitialspace=True)
    template = pd.read_csv('../input/sample_submission_v2.csv')
    
    #read in predictions
    preds_voting = pd.DataFrame()
    preds_voting[target_id] = template[target_id]
    preds_voting['agg_tags'] = ''
    collect_files = []
    collect_preds = []
    for i, row in collection.iterrows():
        tmp = row['submission'].strip()
        collect_files.append(tmp)
        collection.loc[i, 'submission'] = tmp
        collect_preds.append(pd.read_csv(row['submission']))
        preds_voting = preds_voting.merge(collect_preds[-1], how='left', on=target_id)
        preds_voting['agg_tags'] = preds_voting['agg_tags'] + ' ' + preds_voting['tags'] 
        preds_voting.drop('tags', axis=1, inplace=True)
    print('read in {} submissions'.format(len(collect_files)), flush=True) 
    cutoff = ceil(len(collection)/2)
	
    print('using union/majority to make ensemble predictions')
    all_agg_tags = preds_voting['agg_tags'].tolist()
    all_major_tags = []
    
    for i, agg_tags in enumerate(all_agg_tags, start=1):
        labels = []
        if type(agg_tags) is str:
            for x in agg_tags.split(' '):
                labels.append(x)
        
        labels = sorted(list(set(labels)))
        line = ''
        for s in labels:
            if agg_tags.count(s) >= cutoff:
                line += s + ' '
                
        all_major_tags.append(line.strip())
        
        if (i) % 1000 == 0:
            print('processed {:05d} samples'.format(i), flush=True)
    print('processed {:05d} samples'.format(i+1), flush=True)
        
    #dump predicitons
    print('using majority vote (>={:d}) to make ensemble predictions'.format(cutoff))        
    #dump predicitons
    tmstmp = '{}'.format(time.strftime("%Y-%m-%d-%H-%M"))
    filename = 'sumbEns_major_{}.csv'.format(tmstmp)
    subm = pd.DataFrame()
    subm[target_id] = preds_voting[target_id]
    subm['tags'] = all_major_tags #preds_voting['major_tags']
    subm.to_csv(filename, index=False)
