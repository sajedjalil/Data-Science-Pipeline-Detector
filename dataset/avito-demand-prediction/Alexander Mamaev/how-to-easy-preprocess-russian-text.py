# In this kernel I'll show you how easy it is to preprocess the text in Russian.

# You need to install two libraries:
# * nltk - to get russian stopwords
# * pymystem3 - for lemmatization

# download stopwords corpus, you need to run it once
import nltk
nltk.download("stopwords")
#--------#

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation

#Create lemmatizer and stopwords list
mystem = Mystem() 
russian_stopwords = stopwords.words("russian")

#Preprocess function
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text

#Examples    
preprocess_text("Ну что сказать, я вижу кто-то наступил на грабли, Ты разочаровал меня, ты был натравлен.")
#> 'сказать видеть кто-то наступать грабли разочаровывать натравлять'

preprocess_text("По асфальту мимо цемента, Избегая зевак под аплодисменты. Обитатели спальных аррондисманов")
#> 'асфальт мимо цемент избегать зевака аплодисменты обитатель спальный аррондисман'

#Thats all :)
