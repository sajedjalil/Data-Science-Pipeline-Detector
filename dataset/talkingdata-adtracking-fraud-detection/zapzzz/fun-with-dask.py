import pandas as pd
import dask.dataframe as dd
import numpy as np
import time
import gc
from functools import reduce
from collections import OrderedDict
import subprocess
import types
import multiprocessing as mp

import os
print(os.listdir("../input"))

# Define variable because we are using global variables in functions for memory
#  reasons (so we do not copy df's value)
df=''

### Dask group-by partitions + aggregation
# Slow, memory efficient
def sequential_reduce(lst):
    outdf=pd.DataFrame()
    for part_df in lst:
        outdf = outdf.append(part_df, ignore_index=True)
        del part_df
        del lst[0]
        gc.collect()
    return outdf

def safe_rename_cols(df_columns, old, new):
  """
  Valid for both dask and pandas, better than str.replace
  """
  cols = list(df_columns)
  if type(old) == type(new) in [tuple, list]:
    for old_col, new_col in zip(old, new):
        cols[ cols.index(old_col) ] = new_col
  else:
    cols[ cols.index(old) ] = new
  return cols

def to_dtypes(df, dtypes):
    if type(dtypes) in [type, str]:
      df = df.astype( dtypes )
    else: # Assumes dictionary of mapping {column_name : dtype}
      for column, dtype in dtypes.items():
        if column in df.columns:
          df.loc[:, column] = df.loc[:, column].astype(dtype, copy=False)
    return df
    
class dask_groupby_partitions:
  
  def __init__(self):
    self.parquet_count=0
    self.mapped = []
  
  def fn_to_parquet(self, df, groupby_cols, selected_col, operation="count",
                    path=None):
    if path == None:
      path = "{}_{}_{}".format("-".join(groupby_cols), operation, selected_col)
    path = path + "-{}.parquet".format(self.parquet_count)
    getattr(df.groupby(groupby_cols)[selected_col], operation)().reset_index()\
        .to_parquet(path, compression="uncompressed")
    self.parquet_count += 1
    print(gc.collect())
    return 0
  
  def fn_basic(self, df, groupby_cols, selected_col, operation="count"):
      self.mapped.append(
          getattr(df.groupby(groupby_cols)[selected_col], operation)().reset_index() 
          )
      return 0
  
  def fn_apply(self, df, groupby_cols, selected_col, operation):
      self.mapped.append(
          df.groupby(groupby_cols)[selected_col].apply(operation).reset_index() 
          )
      return 0
    
  def fn_dtypes(self, df, groupby_cols, selected_col, operation="count", dtypes=None):
      
      if dtypes != None:
        self.mapped.append(
            to_dtypes(
            getattr(df.groupby(groupby_cols)[selected_col], operation)().reset_index()
                ))
      else:
        self.mapped.append(
          getattr(df.groupby(groupby_cols)[selected_col], operation)().reset_index() 
          )
      gc.collect()
      return 0
  
  @staticmethod
  def queue_input(queue, new_label, result):
    if queue != None:
      print("Returning result to parent process...")
      queue.put( (new_label, result) )
      print("sent")
      
      del result
      gc.collect()
  
  def multiprocess_operations(self, operations):
    """
    Execute multiple mathematical operations on a dataframe in parallel.
    Parameters:
      operations: key-value mapping 
    """
    procs = [] # List to store processes
    queue = mp.Queue()
    for key, operation_kwargs in operations.items():
      # Set new label to `key` if unspecified beforehand
      operation_kwargs["new_label"] = operation_kwargs.get("new_label", key)
      
      operation_kwargs["multiproc_queue"] = queue
      proc = mp.Process(target=dask_groupby_partitions().groupby_partitions,
                        kwargs=operation_kwargs)
      proc.start()
      procs.append(proc)
    
    # Get results from processes
    results = {}
    while len(results) < len(operations):
      k, v = queue.get()
      print("Received data from child process {}".format(k))
      results[k] = v

    print("Received all intermediate results...")

    # Finish intermediate processes
    print("Waiting for child processes to exit...")
    for proc in procs:
      proc.join()
    
    return results
        
  def groupby_partitions(self, groupby_cols, target_col, operation="count", 
                          dtypes=None, merge=False, immediate_dtype_parse=False,
                          to_parquet=False, is_intermediate_operation=False,
                          parallel_reduce=False, **kwargs):
    """
    Groupby dask dataframes by partitions, then aggregate results.
    Suitable for systems with low memory specifications.
    """
    global df
    self.mapped=[] # set mapped to empty list
    rm_df = kwargs.get("rm_df", False)
    
    # Optional parameter to return result to another process
    internal_queue = kwargs.get("multiproc_queue", None)
  
    # Operation to use to groupby dataframes from each partition's groupby results
    aggregate_operation = kwargs.get("aggregate_operation", operation)
  
    
    # Define new label name
    new_label = kwargs.get("new_label", False)
    if not new_label:
      if is_intermediate_operation and \
        type(operation) == str:
          new_label = operation
      else:
          new_label = "{}_{}_{}".format("-".join(groupby_cols), operation, target_col)
    
    # Arguments passed to function in case of recursive call
    temp_kwargs = dict(
          groupby_cols=groupby_cols, target_col=target_col,
          merge=False, immediate_dtype_parse=immediate_dtype_parse,
          to_parquet=to_parquet, is_intermediate_operation=True,
          **kwargs
        )
    if "new_label" in temp_kwargs:
      del temp_kwargs["new_label"]
    
    # Prevent numeric overflow
    if str(df[target_col].dtype) in ["uint8", "uint16", "int8", "int16", "int32"]:
      if str(df[target_col].dtype)[0] == "u":
        new_dtype="uint32"
      else:
        new_dtype="int64"
      df[target_col] = df[target_col].astype( new_dtype )
    
    # Workflow of execution of operation
    # Since most operations have to be adapted because of the computational
    # process (groupby each partition as a batch -> aggregate groupby),
    # these operations are defined below
    if operation not in ["count", "sum", "mean", "var", "ngroups"] \
    and type(operation) != types.FunctionType:
      return NotImplementedError

    if operation == "count":
      aggregate_operation = "sum"
    
    elif operation == "mean":  
      intermediate_operations = dict(
          sum=dict(operation="sum", **temp_kwargs),
          count=dict(operation="count", **temp_kwargs)
      )
      
      intermediate_results = self.multiprocess_operations(intermediate_operations)
        
      # Divide sum / count to get mean
      print("\nFinal aggregation...")
      
      result = intermediate_results["sum"].merge(
                    intermediate_results["count"], on=groupby_cols)
      
      del intermediate_results
      gc.collect()
      
      result[new_label] = result["sum"] / result["count"]
      
      self.queue_input(internal_queue, new_label, result)
      return result

    
    elif operation == "var":
      sum_of_squares_fn = lambda x: np.sum(np.square(x))
      intermediate_operations = dict(
          sum_of_squares=dict(operation=sum_of_squares_fn,
                              aggregate_operation="sum",
                              **temp_kwargs),
          mean=dict(operation="mean", **temp_kwargs)
          )

      intermediate_results = self.multiprocess_operations(intermediate_operations)
      
      # Result: sum of squares / count - (mean squared)
      print("\nFinal aggregation...")
  
      result = intermediate_results["sum_of_squares"].merge(
              intermediate_results["mean"], on=groupby_cols)
      
      del intermediate_results
      gc.collect()
      
      result[new_label] =\
        result["sum_of_squares"] / result["count"] - np.square(result["mean"])
  
      return result
      
    
    elif operation == "ngroups":
      """
      Untested
      """
      # Get unique values of dataframe
      unique_fn = lambda df: df[ groupby_cols ].unique()
      
      unique = dask_groupby_partitions().groupby_partitions(
                    **temp_kwargs, operation=unique_fn)
      
      # Call ngroups on resulting dataframe
      return unique.groupby(groupby_cols).ngroups()
    
    ## Decide what function to run during map_partitions
    fn = self.fn_basic 
  
    if dtypes != None:
      if immediate_dtype_parse: # Set this to True to save some memory during computations
        fn = self.fn_dtypes
      
    # Pass custom function as operation
    if type(operation) == types.FunctionType:
      fn = self.fn_apply
    
    # When memory becomes a real issue, store each grouped-by result into a parquet file
    if to_parquet:
      path = kwargs.get("path", new_label)
      fn = self.fn_to_parquet
    
    ## Map partitions  
    print("Mapping...")
    df.map_partitions(fn, groupby_cols, target_col, operation=operation, 
                      meta=("0", "bool")).compute()
    gc.collect()
    
    if rm_df:
      print("Removing df...")
      del df
      gc.collect()
    
    ## Reduce results of grouped-by batches into a single dataframe
    print("Reducing...")
    if to_parquet: 
      print("Importing generated parquets...")
      mapped_values = lambda: [ dd.read_parquet( temp_parquet) \
                   for temp_parquet in subprocess.getoutput(
                       "find {}*.parquet -maxdepth 0".format(path)).split("\n") ]
    else:
      mapped_values = lambda: self.mapped
      
    if parallel_reduce: # Faster,  may cause spikes in RAM
      gp = reduce(lambda df1, df2: df1.append(df2, ignore_index=True), 
                  mapped_values())
    else:
      gp = sequential_reduce(mapped_values())
    
    del self.mapped
    gc.collect()
  
    ## Execute group-by on results of all batches  
    # You better have enough RAM at this point
    # Otherwise, convert output dataframe to dask and repeat the process until 
    # you have enough RAM
    print("Aggregating...")
    # groupby parameters:
    #  sort=False, faster
    #  as_index=False, so dont have to call .reset_index() afterwards
    result = getattr(
        gp.groupby(groupby_cols, sort=False, as_index=False)[target_col], 
          aggregate_operation)()
  
    ## Post-processing
    result.columns = safe_rename_cols(result.columns, target_col, new_label)
  
    if dtypes != None:
      result = to_dtypes(result, dtypes)
    
    if internal_queue != None:
      # Return result to other process
      self.queue_input(internal_queue, new_label, result)
    
    if not merge:
      return result
    
    elif not rm_df: # Merge with df
      print("Merging with dataframe...")
      df = df.merge( result(), how="left", on=groupby_cols)

### Multi-engine datetime parsing
class datetimeColumnParse:
  def __init__(self, prefix="click_", format_="%Y-%m-%d %H:%M:%S", 
               engine="dask", **kwargs):
    global df
    
    self.orig_col=kwargs.get("orig_col", prefix+"time")
    self.format=format_
    self.prefix=prefix
    
    engines={ "pandas" : pd, "dask" : dd }
    self.engine = engine
    self.eng = engines[engine]
    print("Using engine: {}".format(self.engine))
    
    self.orig_to_datetime()

  def df_column(self, column_name):
    global df
    if self.engine == "pandas": 
      return df.loc[:, column_name]
    elif self.engine == "dask":
      return df[column_name] 
    
  def df_column_set(self, column_name, value):
    global df

    if self.engine == "pandas": 
      df.loc[:, column_name] = value
    elif self.engine == "dask":
      df[column_name] = value

  def orig_to_datetime(self):
    self.df_column_set(self.orig_col, \
      self.eng.to_datetime( self.df_column(self.orig_col), format=self.format)
      )
    gc.collect()
  
  # Split timestamp into day/hour/minute/second 
  def time_unit_split(self, time_units=["day", "hour", "minute", "second"]):
    global df
    
    for time_unit in time_units:
      df[ time_unit ] = getattr(df[self.orig_col].dt, time_unit).astype("uint8")

  def to_unixtime(self):
    dtype="uint32"
    if np.log2(time.time()) >= 32: dtype="uint64" # For the future
    
    self.df_column_set(self.prefix + "unix",
      (self.df_column(self.orig_col).astype(np.int64) // 10**9).astype(dtype)
      )
    
    gc.collect()
      # Split into time periods
  def to_time_periods(self, enum=True, period_start="2017-11-06 16:00:00",
                    period_end="2017-11-09 15:59:00", period_freq="4H"):
    prange = pd.period_range(period_start, period_end, freq=period_freq)
    prange_timestamp = list(prange.to_timestamp())
    
    def get_period_index(x):
      for i, p in enumerate(prange_timestamp):
          if p>x: return i - 1
    #Time passed: 0.386 seconds.

    
    def get_period(x):
      for i, p in enumerate(prange_timestamp):
          if p>x: return prange[i - 1]
    
    """ This could be further optimized if one assumes the rows are sorted
    by date; then just finding the timestamp which comes first after each
    period start/end; and assigning all timestamps in between the start of 
    one period and the start of the next one with the respective period """
    
    apply_kwargs = {}
    if enum:
      self.df_column_set(self.prefix + "time_period",
          self.df_column(self.orig_col).apply(get_period_index,
                                  **apply_kwargs).astype("uint8")
          )
    
    else:
      if self.engine == "dask" : 
        raise NotImplementedError
      self.df_column_set(self.prefix + "time_period",
          self.df_column(self.orig_col).apply(get_period)
          )
    
    gc.collect()
    
    return prange

### Start of processing code
def timer(start=None):
  if start == None:
    return time.time()
  else:
    diff = time.time() - start
    time_units=OrderedDict(dict(
        hours=3600,
        minutes=60,
        seconds=1))
    print("Time passed: ", end='')
    for time_unit in time_units:
      units, modulo = diff // time_units[time_unit], diff % time_units[time_unit]
      diff = modulo
      if time_unit == "seconds": print("{:.3f} {}.".format(units+diff,time_unit))
      elif units > 0: print("{} {}, ".format(int(units), time_unit), end='')

DATADIR="../input/"

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_id'      : 'uint32',
        'is_attributed' : 'bool'
        }

def import_df():
  global df
  df = dd.read_csv(DATADIR+"train.csv", dtype=dtypes)
  
  # Parse datetime
  dtparse=datetimeColumnParse(prefix="", orig_col="click_time")
  dtparse.time_unit_split(["day", "hour"])
  
  for drop_col in ['attributed_time','is_attributed']:
    del df[drop_col]
  gc.collect()

import_df()
df = df[:10**3]
  
# Following statements are just for testing, should not be taken too seriously
groupby_statements = OrderedDict(
    ip_app_count = ( 
        (['ip', 'app'], 'channel', "count"), 
        ),
    ip_app_channel_var_day = (
        (['ip', 'app', 'channel'], "day", "var"),
                              ),
    ip_app_channel_mean_hour = ( 
        (['ip', 'app', 'channel'], "hour", "mean"),
        dict(dtypes={"hour" : "float32"})
                )
    )

# Execute the above statements
# Note: input is a dask dataframe but output is a pandas dataframe
for new_col, args in groupby_statements.items():
  if len(args) == 2:
    kwargs = args[1]
  else:
    kwargs = {}
  args = args[0]
  
  col_dtypes={}
  if 'dtypes' in kwargs.keys():
    col_dtypes = kwargs["dtypes"]
    del kwargs["dtypes"]
  col_dtypes = { **col_dtypes,
      **{ k : dtypes[k] for k in args[0] + [ args[1] ] if dtypes.get(k)}
                  }
  
  print(new_col, col_dtypes)
  
  start_timer = timer()
  result = dask_groupby_partitions().groupby_partitions(
                              *args, **kwargs, dtypes=col_dtypes,
                              immediate_dtype_parse=False,
                               rm_df=False, merge=False, parellel_reduce=True)
  timer(start_timer)
  
  print('', result.head(), result.tail(), sep="\n")
  #result.to_csv("train-grouped_by_{}-unindexed.csv".format(new_col))
  
  del result
  gc.collect()