import pandas as pd
import nltk

def gender_discrimination():
    texts = pd.read_csv("../input/train.csv")
    grouped_by_author = texts.groupby("author")
    word_freq_by_author = nltk.probability.ConditionalFreqDist()
    for name, group in grouped_by_author:
        sentences = group['text'].str.cat(sep=' ')
        sentences = sentences.lower()
        tokens = nltk.tokenize.word_tokenize(sentences)
        frequency = nltk.FreqDist(tokens)
        word_freq_by_author[name] = (frequency)
        
    male = ['he', 'his', 'himself', 'himselves', 'man', 'men', 'him']
    female = ['she', 'hers', 'herself', 'herselves', 'woman', 'women', 'her']
    for author in word_freq_by_author.keys():
        malecount = 0
        femalecount = 0
        for word in male:
            malecount += word_freq_by_author[author].freq(word)
        for word in female:
            femalecount += word_freq_by_author[author].freq(word)
        print(author, malecount * 100., femalecount * 100.)

gender_discrimination()