# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import inspect
import os

import pandas

def get_stats(func):
    def wraps(*args, **kwargs):
        df = func(*args, **kwargs)
        print("nrows : %d" % df.shape[0])
        print("ncolumns : %d" % df.shape[1])
        return df
    return wraps
        
def log_time(func):
    def wraps(*args, **kwargs):
        begin = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        print("Total time taken %fs" % (begin - end))
        return results
    return wraps
    
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
        
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write(function_definition)
        
write_function_to_file(get_stats, "func.py")
write_function_to_file(log_time, "func.py")
