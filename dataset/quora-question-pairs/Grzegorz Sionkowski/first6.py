# Two questions:
# 1. Can you improve on this benchmark?
# 2. Can you beat the score obtained by this example kernel?

from nltk.corpus import stopwords
import pandas as pd


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row[0]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[1]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    #R1 = len(shared_words_in_q1)/len(q1words)
    #R2 = len(shared_words_in_q2)/len(q2words)
    R3 = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    #return (0.5*len(shared_words_in_q1)/len(q1words) + 0.5*len(shared_words_in_q2)/len(q2words))
    return (4*R3 + 0.3692)/5


test = pd.DataFrame.from_csv("../input/test.csv")
stops = set(stopwords.words("english"))
test["is_duplicate"] = test.apply(word_match_share, axis=1, raw=True)
test["is_duplicate"].to_csv("count_words_benchmark.csv", header=True)