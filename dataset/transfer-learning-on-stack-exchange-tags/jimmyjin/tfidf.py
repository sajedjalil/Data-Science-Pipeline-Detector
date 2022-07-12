'''
Created on 2016-12-9

@author: jimmyjin
'''
import re
import csv
import math
from collections import defaultdict, OrderedDict

stop_words = {'a', "a's", 'able', 'about', 'above', 'according', 'accordingly',
              'across', 'actually', 'after', 'afterwards', 'again', 'against',
              "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along',
              'already', 'also', 'although', 'always', 'am', 'among', 'amongst',
              'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone',
              'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear',
              'appreciate', 'appropriate', 'are', "aren't", 'around', 'as',
              'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away',
              'awfully', 'b', 'be', 'became', 'because', 'become', 'becomes',
              'becoming', 'been', 'before', 'beforehand', 'behind', 'being',
              'believe', 'below', 'beside', 'besides', 'best', 'better',
              'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon",
              "c's", 'came', 'can', "can't", 'cannot', 'cant', 'cause',
              'causes', 'certain', 'certainly', 'changes', 'clearly', 'co',
              'com', 'come', 'comes', 'concerning', 'consequently', 'consider',
              'considering', 'contain', 'containing', 'contains',
              'corresponding', 'could', "couldn't", 'course', 'currently', 'd',
              'definitely', 'described', 'despite', 'did', "didn't",
              'different', 'do', 'does', "doesn't", 'doing', "don't", 'done',
              'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight',
              'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially',
              'et', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone',
              'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',
              'f', 'far', 'few', 'fifth', 'first', 'five', 'followed',
              'following', 'follows', 'for', 'former', 'formerly', 'forth',
              'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets',
              'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got',
              'gotten', 'greetings', 'h', 'had', "hadn't", 'happens', 'hardly',
              'has', "hasn't", 'have', "haven't", 'having', 'he', "he's",
              'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter',
              'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him',
              'himself', 'his', 'hither', 'hopefully', 'how', 'howbeit',
              'however', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if',
              'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed',
              'indicate', 'indicated', 'indicates', 'inner', 'insofar',
              'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll",
              "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps',
              'kept', 'know', 'knows', 'known', 'l', 'last', 'lately', 'later',
              'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's",
              'like', 'liked', 'likely', 'little', 'look', 'looking', 'looks',
              'ltd', 'm', 'mainly', 'many', 'may', 'maybe', 'me', 'mean',
              'meanwhile', 'merely', 'might', 'more', 'moreover', 'most',
              'mostly', 'much', 'must', 'my', 'myself', 'n', 'name', 'namely',
              'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither',
              'never', 'nevertheless', 'new', 'next', 'nine', 'no', 'nobody',
              'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing',
              'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often',
              'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only',
              'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our',
              'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own',
              'p', 'particular', 'particularly', 'per', 'perhaps', 'placed',
              'please', 'plus', 'possible', 'presumably', 'probably',
              'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're',
              'really', 'reasonably', 'regarding', 'regardless', 'regards',
              'relatively', 'respectively', 'right', 's', 'said', 'same', 'saw',
              'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing',
              'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves',
              'sensible', 'sent', 'serious', 'seriously', 'seven', 'several',
              'shall', 'she', 'should', "shouldn't", 'since', 'six', 'so',
              'some', 'somebody', 'somehow', 'someone', 'something', 'sometime',
              'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry',
              'specified', 'specify', 'specifying', 'still', 'sub', 'such',
              'sup', 'sure', 't', "t's", 'take', 'taken', 'tell', 'tends', 'th',
              'than', 'thank', 'thanks', 'thanx', 'that', "that's", 'thats',
              'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence',
              'there', "there's", 'thereafter', 'thereby', 'therefore',
              'therein', 'theres', 'thereupon', 'these', 'they', "they'd",
              "they'll", "they're", "they've", 'think', 'third', 'this',
              'thorough', 'thoroughly', 'those', 'though', 'three', 'through',
              'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took',
              'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying',
              'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless',
              'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used',
              'useful', 'uses', 'using', 'usually', 'uucp', 'v', 'value',
              'various', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants',
              'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've",
              'welcome', 'well', 'went', 'were', "weren't", 'what', "what's",
              'whatever', 'when', 'whence', 'whenever', 'where', "where's",
              'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
              'wherever', 'whether', 'which', 'while', 'whither', 'who',
              "who's", 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
              'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder',
              'would', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you',
              "you'd", "you'll", "you're", "you've", 'your', 'yours',
              'yourself', 'yourselves', 'z', 'zero', ''}


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def get_words(text):
    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    return [word.strip().lower() for word in word_split.split(text)]


def process_text1(doc, idf, text):
    tf = OrderedDict()
    word_count = 0.

    for word in get_words(text):
        if word not in stop_words and word.isalpha():

            if word not in tf:
                tf[word] = 0
            tf[word] += 1
            idf[word].add(doc)
            word_count += 1.

    for word in tf:
        tf[word] = tf[word] / word_count

    return tf, word_count


def process_text(doc, dis, text,tags,totoltag,tf,tagwordcount):
    for word in get_words(text):
        if word not in stop_words and word.isalpha():
            for tag in tags:
                if tag.strip()!="":
                    dis[word].add(tag)
                    totoltag.add(tag)
                    tf.setdefault(tag,{})
                    tf[tag].setdefault(word,0)
                    tf[tag][word]+=1
                    tagwordcount.setdefault(tag,0)
                    tagwordcount[tag]+=1


def main():
    data_path = "../input/"
    data_src= ["biology.csv","cooking.csv","crypto.csv","diy.csv","robotics.csv","travel.csv"]
    tf = {}  
    idf ={}
    dis =defaultdict(set)
    totaltag = set()
    tagwordcount = {}
    for path in data_src:
        in_file = open(data_path + path)
        reader = csv.DictReader(in_file)
        for row in reader:
            tags = row["tags"]
            doc = row["id"]
            tags = get_words(tags)
            text = clean_html(row["title"]) + ' ' + clean_html(row["content"])
            process_text(path+str(doc), dis, text,tags,totaltag,tf,tagwordcount)
        
    out_file = open("tf_dis.csv", "w")

    reader = csv.DictReader(in_file)
    writer = csv.writer(out_file)
    writer.writerow(['id', 'tags'])
    
    tagnums = len(totaltag)
    for tag in totaltag:
        for word in tf[tag]:
            tf[tag][word]/=float(tagwordcount[tag])
            idf[word] = math.log(tagnums/(1+len(dis[word])))
            tf[tag][word] *= idf[word]
   # print(tf)
    #print(totaltag)
    test_src = "test.csv"
    # Write predictions
    print("Writing predictions..")
    in_file = open(data_path + test_src)
    reader = csv.DictReader(in_file)
    docs = []

    # Calculate TF and IDF per document
    idf1 = defaultdict(set)
    tf1 = {}
    word_counts = defaultdict(float)

    print("Counting words..")
    for row in reader:
        doc = int(row['id'])
        docs.append(doc)

        text = clean_html(row["title"]) + ' ' + clean_html(row["content"])
        tf1[doc], word_counts[doc] = process_text1(doc, idf1, text)

    in_file.close()

    # Calculate TF-IDF
    nr_docs = len(docs)
    for doc in docs:
        for word in tf1[doc]:
            tf1[doc][word] *= math.log(nr_docs / len(idf1[word]))

    # Write predictions
    print("Writing predictions..")
    for doc in docs:

        # Sort words with frequency from high to low.
        pred_tags = sorted(tf1[doc], key=tf1[doc].get, reverse=True)[:3]
        score = {}
        for word in pred_tags:
            if word not in stop_words and word.isalpha():
                for tag in totaltag:
                    score.setdefault(tag,0.0)
                    if tag not in tf:
                        continue
                    if word not in tf[tag]:
                        continue
                   # print(tf[tag][word])
                    score[tag] += tf[tag][word]
        score = list(score.items())
#print(score)
        score.sort(key = lambda x:x[1],reverse = True)
        result = [x for (x,y) in score]
        writer.writerow([doc, " ".join(sorted(result[:3]))])
        #print(doc)
        
        # Write predictions
    print("task finish!")
    in_file.close()
    out_file.close()


if __name__ == "__main__":
    print("Starting program.")
    main()
