import numpy as np
import pandas as pd
import os.path
import subprocess



def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def skiplines_train(remaining_Lines):
    lines = file_len('../input/train.csv')
    
    #generate list of lines to skip
    skiplines = np.random.choice(np.arange(1, lines), size=lines-1-remaining_Lines, replace=False)

    #sort the list
    return np.sort(skiplines)
    
    
def main():
    
    datapath = '../input/'
    
    dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'         
        }
    
    cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    
    print('Choosing lines')
    
    remaining_Lines = 20000000 #20.000.000
    skiplines = skiplines_train(remaining_Lines)
    
    print('Lines selected. Now importing Train dataset')
    
    train = pd.read_csv(os.path.join(datapath, 'train.csv'), usecols=cols, dtype=dtypes, parse_dates=['click_time'], skiprows=skiplines)
    
    print('Train imported')
    
    
    train.to_csv("train_20kk_Lines.csv", index=False)
    
    print(train.info())

    
if __name__ == '__main__':
    main()
    