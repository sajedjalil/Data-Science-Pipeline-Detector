import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

gifts = pd.read_csv('../input/gifts.csv')
gifts.info()
gifts.head()
gifts.describe()

# All gift locations
gifts.plot.scatter('Longitude', 'Latitude', alpha=0.5, s=10, color='blue')
plt.title('Gift Locations; N = {:,} ; alpha ={;,}'.format(len(gifts)))
plt.show()
plt.savefig('All-gift-locations.png')

# All gift locations -2
gifts.plot.scatter('Longitude', 'Latitude', alpha=0.5, s=5, color='red')
plt.title('Gift Locations; N = {:,}'.format(len(gifts)))
plt.show()
plt.savefig('All-gift-locations2.png')

# All gift locations -3
gifts.plot.scatter('Longitude', 'Latitude', alpha=0.5, s=1, color='blue')
plt.title('Gift Locations; N = {:,}'.format(len(gifts)))
plt.show()
plt.savefig('All-gift-locations3.png')

# All gift locations -4
gifts.plot.scatter('Longitude', 'Latitude', alpha=0.5, s=0.5, color='blue')
plt.title('Gift Locations; N = {:,}'.format(len(gifts)))
plt.show()
plt.savefig('All-gift-locations4.png')