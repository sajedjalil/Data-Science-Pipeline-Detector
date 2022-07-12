import pandas as pd
import matplotlib.pyplot as plt

def plot_all_locations(grouped_data, place_id):
  print('Plotting data....');
  place = grouped_data.get_group(place_id)
  plt.plot(place.x, place.y, 'ro', place.x.mean(), place.y.mean(), 'bs')
  plt.show();

print('Reading data....')
all_data = pd.read_csv('../input/train.csv')
grouped_data = all_data.groupby(['place_id'])

place_id = all_data.sample(n=1).place_id.values[0]
plot_all_locations(grouped_data, place_id)

