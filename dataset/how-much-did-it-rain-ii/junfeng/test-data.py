
import pandas as pd
import numpy as np

import sys
from datetime import datetime

print(sys.version)


# change the location of the downloaded test file as necessary.
infile="../input/test.csv"
#infile="kaggle/sample.csv"
outfile="sample_solution.csv"

# Make sure you are using 64-bit python.
if sys.maxsize < 2**32:
    print("You seem to be running on a 32-bit system ... this dataset might be too large.")
else:
    print("Hurray! 64-bit.")

print(datetime.now())
# read file
alldata = pd.read_csv(infile)
# alldata = alldata.set_index('Id')


# In[8]:

print(datetime.now())


# In[9]:

def marshall_palmer(ref, minutes_past):
    print("Estimating rainfall from {0} observations".format(len(minutes_past)))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum


# In[10]:

# each unique Id is an hour of data at some gauge
def myfunc(hour):
    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est


# In[19]:

alldata.set_index("Id")


print(datetime.now())


# In[31]:

groups = alldata.groupby(by=["Id"])

print(datetime.now())
# In[ ]:

estimates = groups.apply(myfunc)


# In[37]:

print(datetime.now())


# ```python
# # this writes out the file, but there is a bug in dask
# # where the column name is '0': https://github.com/blaze/dask/pull/621
# estimates = alldata.groupby(alldata.index).apply(myfunc, columns='Expected')
# estimates.to_csv(outfile, header=True)
# ```

# In[63]:

estimates.name = "Expected"


# In[65]:

estimates.to_csv(outfile, header=True)

print(datetime.now())
