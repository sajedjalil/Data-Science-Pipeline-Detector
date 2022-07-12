# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import collections
import csv
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Every possible `substring` of a list, where order is maintained.
def allPossibleSublists(superList):
    return [(i, j, superList[i:j]) for i in range(len(superList))
                                    for j in range(i + 1, len(superList) + 1)]

class BasicProbabilities:

    def __init__(self):
        # wordSentiment['hello'] = (count, estimatedSentiment)
        self.positiveCount    = collections.defaultdict(int)
        self.negativeCount    = collections.defaultdict(int)
        self.nonPositiveCount = collections.defaultdict(int)
        self.nonNegativeCount = collections.defaultdict(int)

        self.positiveVocabulary = set()
        self.negativeVocabulary = set()

        self.wordCost = 0.0000005
        self.agreementWeight = 1
        self.nonAgreementWeight = -0.6
        self.nonDisagreementWeight = -0.4
        self.disagreementWeight = -0

        self.ignoreStopWords = False
        self.ignoreLinks = True
        self.shortenElongations = False
        self.englishWords = set()

        # The various sources of stop words
        pronouns = {"i", "me", "us", "you", "she", "her", "he", "him", "it", "we", "us", "they", "them", "this", "these"}
        # Source: https://en.wikipedia.org/wiki/English_personal_pronouns
        copulae = {"be", "is", "am", "are", "being", "was", "were", "been"}
        # Source: https://en.wikipedia.org/wiki/Copula_(linguistics)#English
        conjunctions = {"for", "and", "nor", "but", "or", "yet", "so", "that", "which", "because", "as", "since", "though", "while", "whereas"}
        # Source: https://en.wikipedia.org/wiki/Conjunction_(grammar)
        others = {"a", "the"}

        self.stopWords = others.union(pronouns).union(copulae).union(conjunctions)

        # Setup
        if self.shortenElongations:
            dictionaryFile = open("enable1.txt")
            for line in dictionaryFile.readlines():
                self.englishWords.add(line.strip())
            dictionaryFile.close()

    # Convert word to lowercase and strip all non-letter characters/
    def clean(self, word):
        # Replace non-letter characters with spaces.
        letters = set("abcdefghijklmnopqrstuvwxyz")

        if self.ignoreLinks:
            # Removes obvious links.
            if "http://" in word:
                return ''

            # Removes less-obvious links. Anything with 3 pairs of >= 2 letters
            # separated by periods gets removed. i.e. www.test.com
            if re.match(r"[a-z][a-z]+\.[a-z][a-z]+\.[a-z][a-z]+", word):
                return ''

        word = ''.join([char for char in word.lower() if char in letters])

        if self.ignoreStopWords and word in self.stopWords:
            return ''

        if self.shortenElongations:
            word = self.unElongate(word)

        return word


    # Create a list of words corresponding to the text split on spaces. Note: Some
    # of the tokens may be empty, but are left to retain the correct indices.
    def tokenize(self, text):
        cleanedWords = [self.clean(word) for word in text.split()]

        return cleanedWords


    def unElongate(self, word):
        shortestWord = []

        # First try reducing to one copy.
        for index, letter in enumerate(word):
            # Only keep the current letter if the last one was different.
            if index != 0 and letter == word[index-1]:
                    continue
            shortestWord.append(letter)

        shortestWord = ''.join(shortestWord)

        # Check if this is an english word
        if shortestWord in self.englishWords:
            return shortestWord
        else:
             # Otherwise, try reducing to 2 copies
            shorterWord = []

            for index, letter in enumerate(word):
                # Only keep the current letter if the last 2 were different.
                if index > 2 and letter == word[index-1] and letter == word[index-2]:
                    continue

                shorterWord.append(letter)

            shorterWord = ''.join(shorterWord)

            return shorterWord


    def fit(self, train):
        # One of the training examples has no text
        train.dropna()

        for index, row in train.iterrows():
            if type(row['text']) != str:
                continue

            sentiment = row['sentiment']
            text = row['text'].strip()
            selectedText = row['selected_text'].strip()

            selectionStart = text.find(selectedText)
            selectionEnd = selectionStart + len(selectedText)
            notSelectedText = (text[:selectionStart].strip() + text[selectionEnd:]).strip()

            notSelectedWordList = self.tokenize(notSelectedText)
            selectedWordsList = self.tokenize(selectedText)

            if sentiment == 'neutral':
                continue
                # notSelectedWordList = selectedWordsList

            for word in selectedWordsList:
                if word != '':
                    if sentiment == 'positive':
                        self.positiveVocabulary.add(word)
                        self.positiveCount[word] += 1
                    elif sentiment == 'negative':
                        self.negativeVocabulary.add(word)
                        self.negativeCount[word] += 1

            for word in notSelectedWordList:
                if word != '':
                    # Keep the nonNegatives and nonPositives separate
                    # so that we can weight them individually when
                    # calculating probabilities
                    if sentiment == 'positive':
                        self.nonPositiveCount[word] += 1
                        self.positiveVocabulary.add(word)
                    elif sentiment == 'negative':
                        self.nonNegativeCount[word] += 1
                        self.negativeVocabulary.add(word)


    def positiveProbability(self, word):
        # Previously, this function just calculated a positiveAmount and a
        # neutral amount, added them up, and divided by the length of the
        # positive vocabulary.

        # Here, it's been modified to a) separate "neutrals" into nonPositives
        # and nonNegatives (nonAgreements and nonDisagreements), individually weighted,
        # and b) included a weighted negativeAmount (disagreement). More or less, this part
        # just gives more model flexibility

        positiveAmount = self.positiveCount[word] * self.agreementWeight
        nonPositiveAmount = self.nonPositiveCount[word] * self.nonAgreementWeight
        nonNegativeAmount = self.nonNegativeCount[word] * self.nonDisagreementWeight
        negativeAmount = self.negativeCount[word] * self.disagreementWeight
        totalCount = self.positiveCount[word] + self.nonPositiveCount[word] + self.negativeCount[word] + self.nonNegativeCount[word]
        if totalCount == 0:
            return 0

        # Rather than dividing by the length of the positive vocabulary, why not
        # divide by the number of times we've seen the word in total? In other words,
        # if we've seen the word 100 times, and it was marked as positive every single
        # time (occurred in a positive sentiment and was part of the selected text),
        # it should be given a 100% probability. If it was sometimes not selected,
        # or included in a non-positive sentiment (selected or not), then we should
        # account for that in the numerator (weighted) as well as the denominator (unweighted).
        # With a few tests and slight tweaking, we've achieved 0.69068 average 10-fold cross
        # validation accuracy (std. dev. 0.01184).
        return (positiveAmount + nonPositiveAmount + nonNegativeAmount + negativeAmount) / totalCount


    def negativeProbability(self, word):
        positiveAmount = self.positiveCount[word] * self.disagreementWeight
        nonPositiveAmount = self.nonPositiveCount[word] * self.nonDisagreementWeight
        nonNegativeAmount = self.nonNegativeCount[word] * self.nonAgreementWeight
        negativeAmount = self.negativeCount[word] * self.agreementWeight
        totalCount = self.positiveCount[word] + self.nonPositiveCount[word] + self.negativeCount[word] + self.nonNegativeCount[word]
        if totalCount == 0:
            return 0
        #return (negativeAmount + neutralAmount) / len(self.negativeVocabulary)
        return (positiveAmount + nonPositiveAmount + nonNegativeAmount + negativeAmount) / totalCount


    def selectSubstring(self, text, sentiment):
        # For neutral, just select all of the text.
        if sentiment == 'neutral':
            return text

        wordList = self.tokenize(text)

        if sentiment == 'positive':
            probabilities = list(map(self.positiveProbability, wordList))
        if sentiment == 'negative':
            probabilities = list(map(self.negativeProbability, wordList))

        selected = None
        bestScore = 0

        for i, j, wordScorePairs in allPossibleSublists(list(zip(wordList, probabilities))):
            wordScores = [p for (_, p) in wordScorePairs]
            words      = [w for (w, _) in wordScorePairs if w != '']

            score = sum(wordScores) - len(words) * self.wordCost

            if score > bestScore:
                selected = (i,j)
                bestScore = score

        if selected is None:
            return text
        else:
            start, end = selected
            selectedString = ' '.join(text.split()[start:end])

            return selectedString


    def predict(self, test):
        predictions = pd.DataFrame([], index=test.index, columns=['selected_text'])

        for index, row in test.iterrows():
            text = row['text']
            sentiment = row['sentiment']

            predictedSubstring = self.selectSubstring(text, sentiment)
            predictions.at[index, 'selected_text'] = predictedSubstring

        return predictions

def generateSubmission(onKaggle=False):
    # Files constants
    if onKaggle:
        basePath = '/kaggle'
    else:
        basePath = '.'  # Locally

    cachePath = basePath + '/working/'
    inputPath = basePath + '/input/tweet-sentiment-extraction/'
    testPath = inputPath + 'test.csv'
    trainingPath = inputPath + 'train.csv'

    model = BasicProbabilities()

    # Load the training data
    train = pd.read_csv(trainingPath, index_col=0)
    model.fit(train)

    # Load the testing data
    test = pd.read_csv(testPath, index_col=0)
    predictions = model.predict(test)

    # Create submission file
    predictions.to_csv('submission.csv')


if __name__ == "__main__":
    generateSubmission(onKaggle=True)
