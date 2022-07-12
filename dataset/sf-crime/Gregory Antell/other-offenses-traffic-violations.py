import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

z = zipfile.ZipFile('../input/train.csv.zip')
training_df = pd.read_csv(z.open('train.csv'))
training_df.head(10)

otheroffense_mask = training_df['Category'] == 'OTHER OFFENSES'
otheroffenses_df = training_df[otheroffense_mask]

# Word Cloud of OTHER OFFENSES
otheroffenses_words = ' '.join(list(otheroffenses_df.Descript)).replace(',','')
wordcloud = WordCloud(
    stopwords=STOPWORDS,
    background_color='black',
    width=1600,
    height=800
    ).generate(otheroffenses_words)

plt.figure( figsize=(16,10)) 
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('otherwordcloud.png')
plt.show()