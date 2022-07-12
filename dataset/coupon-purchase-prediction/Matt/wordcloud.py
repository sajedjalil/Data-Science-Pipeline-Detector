import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import pandas as pd

h         = pd.read_csv('../input/coupon_area_train.csv', encoding='utf-8')
lb        = LabelEncoder()
ids       = lb.fit_transform(h.SMALL_AREA_NAME.values).astype(str)
wordcloud = WordCloud().generate(" ".join(['AC' + i for i in ids]))

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('small_area_names.png')
