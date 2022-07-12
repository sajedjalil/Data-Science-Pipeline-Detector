import numpy as np 
import pandas as pd
from  tqdm import tqdm
import pickle
import gc
gc.enable()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, GRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model, load_model

COMP = 'jigsaw-toxic-comment-classification-challenge'
EMBEDPATH = 'fasttext-crawl-300d-2m/crawl-300d-2M.vec'
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAXWORDS = 100000
SEQLEN = 256

def load_data():
    train = pd.read_csv(f"../input/{COMP}/train.csv")
    test = pd.read_csv(f"../input/{COMP}/test.csv")
    submission = pd.read_csv(f'../input/{COMP}/sample_submission.csv')
    return train,test,submission
    
#loading embeddings
def get_vector(word,*line):
    return word,np.array(line,dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_vector(*line.strip().split(' ')) for line in f)
        

#function for preprocessing data
def preprocess(data):
    
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data
 
def convert_to_sequence(train,test):
    train['comment_text'] = preprocess(train['comment_text'])
    test['comment_text'] = preprocess(test['comment_text'])
    df = pd.concat([train[['id','comment_text']],test],axis=0)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['comment_text'])
    
    xtrain = pad_sequences(tokenizer.texts_to_sequences(train['comment_text']),maxlen=SEQLEN)
    xtest = pad_sequences(tokenizer.texts_to_sequences(test['comment_text']),maxlen=SEQLEN)
    ytrain = train[CLASSES].values
    
    return xtrain,xtest,ytrain,tokenizer
    
#building embedding matrix
def build_embedding_matrix(wordindex,dimensions):
    embeddings_index = load_embeddings(f"../input/{EMBEDPATH}")
    embeddings_matrix = np.zeros(dimensions)
    
    for word,i in tqdm(wordindex.items()):
        if i >= dimensions[0] : continue
        vector = embeddings_index.get(word)
        if vector is not None: embeddings_matrix[i] = vector
    
    del embeddings_index
    gc.collect()
    return embeddings_matrix    

#model
def train_model(seqlen,embeddings_matrix,xtrain, ytrain):

    inp = Input(shape=(seqlen,))
    x = Embedding(embeddings_matrix.shape[0],embeddings_matrix.shape[1],
                                weights=[embeddings_matrix],
                                input_length=seqlen,
                                trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)   
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

    avg_pool1 = GlobalAveragePooling1D()(x)
    max_pool1 = GlobalMaxPooling1D()(x)     

    x = concatenate([avg_pool1, max_pool1])

    out = Dense(6, activation='sigmoid')(x)

    
    model = Model(inp, out)
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(xtrain, ytrain, batch_size=32, epochs=2)
    
    return model
    
def submit_preds(model,xtest,submission):
    ytest = model.predict([xtest], batch_size=1024, verbose=1)
    submission[CLASSES] = ytest
    submission.to_csv('submission.csv',index=False)
    
    return

def save(model,tokenizer):
    model.save('model.h5')
    with open('tokenizer.pickle','wb') as f:
        pickle.dump(tokenizer,f)
        
    
def main():
    #load data
    train,test,submission = load_data()
    #tokenize and convert to sequence
    xtrain,xtest,ytrain,tokenizer = convert_to_sequence(train,test)
    #create embeddings matrix
    embeddings_matrix = build_embedding_matrix(tokenizer.word_index,(MAXWORDS,300))
    #train model
    model = train_model(SEQLEN,embeddings_matrix,xtrain,ytrain)
    #submit predictions
    submit_preds(model,xtest,submission)
    #save model
    save(model,tokenizer)

if __name__ == "__main__":
    main()

