import pandas as pd
import matplotlib.pyplot as plt

gifts = pd.read_csv('../input/gifts.csv')

plt.hist(gifts['Weight'].tolist(), bins=50)
plt.title('Gifts Weight Distribution by Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')

plt.show()
plt.savefig('Gift_Distributions_by_Weight.png')