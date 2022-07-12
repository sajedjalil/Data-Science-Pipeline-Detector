#import dask.dataframe as dd
import pandas as pd
import numpy as np

infile="../input/test.csv"
outfile="sample_solution.csv"

# read file
alldata = pd.read_csv(infile)

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

# each unique Id is an hour of data at some gauge
def myfunc(hour):
    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est


groups = alldata.groupby(by=["Id"])
alldata = alldata.set_index('Id')
estimates = groups.apply(myfunc)
estimates.name = "Expected"
estimates.to_csv(outfile, header=True)