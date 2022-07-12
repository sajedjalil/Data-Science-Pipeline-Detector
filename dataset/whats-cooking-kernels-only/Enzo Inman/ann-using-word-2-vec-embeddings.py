import gensim #For Word 2 Vec
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau

train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')

#convert everything to lower
train_df["ingredients"]=[[word.lower() for word in line] for line in train_df["ingredients"]]
test_df["ingredients"]=[[word.lower() for word in line] for line in test_df["ingredients"]]

##convert everything to lower,split on spaces, remove stop words

#train
stop_words = set(stopwords.words('english'))
train_df["ingredients"]=[[word.lower().split() for word in line] for line in train_df["ingredients"]]
train_df["ingredients"]=[
        [word for sub in line for word in sub if word not in stop_words]
        for line in train_df["ingredients"]
        ]

test_df["ingredients"]=[[word.lower().split() for word in line] for line in test_df["ingredients"]]
test_df["ingredients"]=[
        [word for sub in line for word in sub if word not in stop_words]
        for line in test_df["ingredients"]
        ]
#Word to Vec
all_ing_lol=train_df["ingredients"].tolist()+test_df["ingredients"].tolist()
w2v=gensim.models.Word2Vec(all_ing_lol,size=700,window=30,min_count=3,iter=80)
w2v_vec_dict=dict(zip(w2v.wv.index2word,w2v.wv.vectors))

#remove all non encoded words
encoded_words=list(w2v_vec_dict.keys())

#convert data to one encoded row per recipe by averging
train_ing_clean=[np.array([w2v_vec_dict[word] for word in l if word in encoded_words]) for l in train_df["ingredients"]]
avg_ing_train=[np.mean(x,axis=0) for x in train_ing_clean]

test_ing_clean=[np.array([w2v_vec_dict[word] for word in l if word in encoded_words]) for l in test_df["ingredients"]]
avg_ing_test=[np.mean(x,axis=0) for x in test_ing_clean]

##Lets model
x_train=pd.DataFrame(avg_ing_train)
x_test=pd.DataFrame(avg_ing_test)
y=pd.get_dummies(train_df["cuisine"])
y_order=list(y.columns)
X_train,X_val,y_train,y_val=train_test_split(x_train,y,test_size=.05,random_state=42)
model=Sequential()
model.add(Dense(700, activation = "relu",input_shape=(700,)))
model.add(Dropout(.1))
model.add(Dense(500, activation = "relu"))
model.add(Dropout(.2))
model.add(Dense(256, activation = "relu"))
model.add(Dropout(.2))
model.add(Dense(20, activation = "softmax"))

##Compile
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

#helps converge faster
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 30
batch_size=400

#fit model
history=model.fit(
        X_train,y_train,batch_size=batch_size,
        validation_data=(X_val,y_val),epochs = epochs,callbacks=[learning_rate_reduction]
        )


#Write Out Predictions
classes=model.predict(x_test)
preds=np.argmax(classes,axis = 1)
preds=[y_order[x] for x in preds]
pd.DataFrame({"id":test_df["id"].tolist(),"cuisine":preds}).to_csv("avg_w2v.csv",index=False)

        
        
        