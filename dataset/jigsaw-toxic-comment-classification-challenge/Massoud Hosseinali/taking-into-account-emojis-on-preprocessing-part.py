# I saw this preprocessing part in another kernel, here is a very brief analysis on the importance of taking into account the EMOJIS
# since they are indicator of the commentator's mood to some extent it might be good to add them into analysis
# there are about 110500 instances of these replacements in training data set and about 100k in test dataset
# would be glad to hear your opinion on this
# ***************  BELOW IS THE OUTPUT IF YOU DONT WISH TO RUN THE CODE YOURSELVES ******************
# on training data set: 
# Total Number of Comments analyzed: 159571
# Total Number of  words analyzed: 10734904
# Total Number of replacements: 110561
# on test data set: 
# Total Number of Comments analyzed: 153164
# Total Number of  words analyzed: 9436549
# Total Number of replacements: 100966
# ***************************************************************************************************

import pandas as pd

Training_data = pd.read_csv('../input/train.csv')
Test_dataset = pd.read_csv('../input/test.csv')

repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

repl_keys_list = list(repl.keys())

comments = Training_data['comment_text'].tolist()

total_num_of_replsments = 0
total_num_of_comments = 0
total_num_of_words = 0
for comment in comments:
    total_num_of_comments += 1
    split_text = comment.lower().split()
    for word in split_text:
        total_num_of_words += 1
        if word in repl_keys_list:
            total_num_of_replsments += 1

print('on training data set: ')
print('Total Number of Comments analyzed: {}'.format(total_num_of_comments))
print('Total Number of  words analyzed: {}'.format(total_num_of_words))
print('Total Number of replacements: {}'.format(total_num_of_replsments))

comments = Test_dataset['comment_text'].tolist()

total_num_of_replsments = 0
total_num_of_comments = 0
total_num_of_words = 0
for comment in comments:
    total_num_of_comments += 1
    split_text = comment.lower().split()
    for word in split_text:
        total_num_of_words += 1
        if word in repl_keys_list:
            total_num_of_replsments += 1

print('on test data set: ')
print('Total Number of Comments analyzed: {}'.format(total_num_of_comments))
print('Total Number of  words analyzed: {}'.format(total_num_of_words))
print('Total Number of replacements: {}'.format(total_num_of_replsments))