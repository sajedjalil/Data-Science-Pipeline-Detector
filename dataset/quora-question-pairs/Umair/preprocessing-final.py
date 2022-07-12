import pandas as pd
from keras.preprocessing import text,sequence
import pandas as pd
import numpy as np
from tqdm import tqdm

#Reading Train and Test Dataset
data_train=pd.read_csv('../input/quora-question-pairs/train.csv')
data_test=pd.read_csv('../input/quora-question-pairs/test.csv')

#Extracting data from dataset
train_question1=list(data_train.question1.values.astype(str))
train_question2=list(data_train.question2.values.astype(str))
test_question1=list(data_test.question1.values.astype(str))
test_question2=list(data_test.question2.values.astype(str))
y_train=data_train.is_duplicate.values

#Tokenizing each question
tokenizer= text.Tokenizer()
#Creating a dictionary to map each unique word in the whole corpus with a unique number
tokenizer.fit_on_texts(train_question1 + train_question2 + test_question1 + test_question2)

#Total vocabulary
word_index = tokenizer.word_index
#Setting maximum length of a question to 25,pruning other words
max_len=25

#Converting each sentence to sequence of integers using dictionary created above

x1 = tokenizer.texts_to_sequences(data_train.question1.values.astype(str))
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tokenizer.texts_to_sequences(data_train.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

z1= tokenizer.texts_to_sequences(data_test.question1.values.astype(str))
z1 = sequence.pad_sequences(z1, maxlen=max_len)

z2 = tokenizer.texts_to_sequences(data_test.question2.values.astype(str))
z2 = sequence.pad_sequences(z2, maxlen=max_len)

#Opening Glove word vector file and extracting embedded vector for each word
embeddings_index = {}
f = open('../input/glove840b300dtxt/glove.840B.300d.txt',encoding="utf8")
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s Glove word vectors.' % len(embeddings_index))

#Building embedding matrix only for those words which are present in  our vocabulary(word_index)
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
		
		
		
q1_train=x1
q2_train=x2
q1_test=z1
q2_test=z2
#Saving each numpy array in a folder 'preprocessed_data'
np.save(open('q1_train.npy', 'wb'), q1_train)
np.save(open('q2_train.npy', 'wb'), q2_train)
np.save(open('q1_test.npy', 'wb'), q1_test)
np.save(open('q2_test.npy', 'wb'), q2_test)
np.save(open('word_index.npy','wb'),word_index)
np.save(open('y_train.npy', 'wb'), y_train)
np.save(open('embedding_matrix.npy', 'wb'), embedding_matrix)
print("PreProcessing Done")		