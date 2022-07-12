import pandas as pd
import numpy as np

app_events = pd.read_csv('../input/app_events.csv')
events = pd.read_csv('../input/events.csv')

# app_events
# from 1GB to 433MB
app_events.is_active = app_events.is_active.astype(np.int8)
app_events.is_installed = app_events.is_installed.astype(np.int8)
app_events.event_id = app_events.event_id.astype(np.int32)

# events
# from 124MB to 86MB
events.event_id = events.event_id.astype(np.int32)
events.longitude = events.longitude.astype(np.float32)
events.latitude = events.latitude.astype(np.float32)
