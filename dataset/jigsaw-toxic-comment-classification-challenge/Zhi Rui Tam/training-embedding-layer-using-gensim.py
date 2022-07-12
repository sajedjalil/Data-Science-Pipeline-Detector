from gensim.models import word2vec
import glob
import pandas as pd
import string,re
import logging
from tqdm import tqdm
import pickle

try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

table = maketrans("","",string.punctuation)

def train_model(dimension):
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    list_sentences_train = train["comment_text"].fillna("UNK").values
    list_sentences_test = test["comment_text"].fillna("UNK").values
    output = open('text_seg.txt', 'w')
    train_list = []
    for sentence in tqdm(list_sentences_train):
        sentence = sentence.translate(table)
        train_list.append(sentence)
        output.write(sentence)
    test_list = []
    pickle.dump(train_list,open("train_sentence_list.pkl","wb"))
    for sentence in tqdm(list_sentences_test):
        sentence = sentence.translate(table)
        test_list.append(sentence)
        output.write(sentence)
    pickle.dump(test_list,open("test_sentence_list.pkl","wb"))
    
    output.close()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("text_seg.txt")
    model = word2vec.Word2Vec(sentences, size=dimension,iter = 10,sg=1,workers=4)
    model.save("med"+str(dimension)+"_skipgram.model.bin")

if __name__ == "__main__":
    train_model(300)