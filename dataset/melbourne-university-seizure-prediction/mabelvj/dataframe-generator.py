import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from scipy.io import loadmat

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


def df_generator(data_loc = '../input', dataset_group = 'train', **kwargs):
    """ Returns DataFrame containing all information the available in from .t files 
        for a given set of folders.
    
    Args:
        data_loc: relative or absolute path to parent folder containing the datasets.
        dataset_group: prefix of the folders containing the information. 
            In this case: 'train' or 'test'. If a determined folder is to be examined
            just set dataset_group to the name of the folder, i.e. 'train_1'.
        N_samples (optional): number of random samples to take from each pation
    Returns:
        A DataFrame with columns containing information about patient, training sequence, 
        interictal information, and all the data for each electrode.
        
        The columns of the dataframe are: =['patient', 'training_seq', 'interict', 
        'nSamplesSegment', 'sequence', 'data', 'channelIndex', 'iEEGsamplingRate']
        
        Example:
        >> print (df.head())
            channelIndex                                               data  \
            0           1.0  [8.81141, 12.8114, 12.8114, 3.81141, -10.1886,...   
            1           2.0  [57.3984, 46.3984, 47.3984, 43.3984, 46.3984, ...   
            2           3.0  [49.5586, 44.5586, 41.5586, 40.5586, 38.5586, ...   

               iEEGsamplingRate  interict  nSamplesSegment  patient  sequence  \
            0             400.0       0.0         240000.0      1.0       5.0   
            1             400.0       0.0         240000.0      1.0       5.0   
            2             400.0       0.0         240000.0      1.0       5.0   
       
               training_seq  
            0         503.0  
            1         503.0  
            2         503.0  
    """    
    N_samples =  None
    for key in kwargs:
        if key == 'N_samples':
            N_samples =  kwargs[key]
    train_folders = [s for s in next(os.walk(data_loc))[1] if dataset_group in s]
    column_names = ['patient', 'training_seq', 'interict', 'nSamplesSegment', 'sequence', 'data', 
        'channelIndex', 'iEEGsamplingRate']
    dtype= [('patient', np.uint8), ('trainin_seq', np.uint16), ('interict', np.uint8),
        ('nSamplesSegment', np.uint32), ('sequence', np.uint8), ('data', np.float16),
        ('channelIndex', np.uint8), ('iEEEsamplingRate', np.uint16)]    
    df = pd.DataFrame(data = np.empty(len(column_names), dtype= dtype), columns = column_names)
    for k, train_folder in enumerate(train_folders): #iterates over the train_folders
        file_path = os.path.join(data_loc,train_folder) #change to test if desired
        file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        if N_samples is None: #iterate over all the files in the folder per user
            file_names_list = sorted(file_names, key = lambda x: (int(x[:-4].split('_')[0]), 
                int(x[:-4].split('_')[1])))
        else: # sampling each user 
            file_names_list = random.sample(file_names, N_samples)
        for i, file_name in enumerate(file_names_list): #iterates over files
            mat = loadmat(os.path.join(file_path, file_name), verify_compressed_data_integrity= False)
            # verify_compressed_data_integrity = False to avoid corrupted .mat files
            mdata = mat['dataStruct']
            mtype = mdata.dtype
            ndata = {n: mdata[n][0,0] for n in mtype.names}
            if 'test' in dataset_group: #interictal information removed
                patient, training_seq = map(int,file_name.split('.')[0].split('_')) #convert to int
                interictal = None
                sequence = None #sequence does not appear in test files
            else:
                patient, training_seq, interictal = map(int,file_name.split('.')[0].split('_'))
                sequence= ndata['sequence'][0]
            for l, channel in  enumerate(ndata['channelIndices'][0]): #if test interictal= []
                df= df.append(pd.DataFrame({'patient': patient, 'training_seq': training_seq, 'interict': interictal,
                    'nSamplesSegment': ndata['nSamplesSegment'][0], 'sequence': sequence,
                    'data': [ndata['data'][:, l]], 'channelIndex': channel, 'iEEGsamplingRate': ndata['iEEGsamplingRate'][0][0]}),
                    ignore_index=True)            

    return df

train_df = df_generator(data_loc = '../input', dataset_group = 'train_1', N_samples = 10)
print (train_df.head())
test_df = df_generator(data_loc = '../input', dataset_group = 'test_1', N_samples = 10)
print (test_df.head())
#print (test_df.head())    
