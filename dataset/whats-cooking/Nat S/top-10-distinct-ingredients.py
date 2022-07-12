import numpy as np
import pandas as pd
from collections import Counter
import operator

train = pd.read_json('../input/train.json')
ingredients = train['ingredients'].apply(lambda x:','.join(x))
cuisines = train['cuisine']

out = '<html>\n'

#ingredients are a list of comma seperated ingredient strings
#by joining them with a comma and then splitting on a comma we get a list of all the ingredients
#Counter then returns a dict of all of the ingredients and the number of times they occur
total = Counter(','.join(ingredients).split(','))
for cuisine in cuisines.unique():
    ing = Counter(','.join(ingredients[cuisines == cuisine]).split(','))
    
    #We create a new dict of all the keys from ingredient counter and give it a 'uniqueness' score
    #This score is the frequency of an ingredient within recipes in a certain cuisine 
    #multiplied by the relative frequency of that ingredient to the frequency in the entire recipe corpus.
    #This gives us a measure of how unique an ingredient is to a certain cuisine, but isn't 
    #distorted by low frequency, highly unique ingredients
    ing_score = {k:(float(ing[k])/len(ing))**2/(float(total[k])/len(total)) for k in ing.keys()}
                
    out += "<b>"+cuisine+"</b>\n<ol type=\"1\">\n"
    for item in sorted(ing_score.items(), key = lambda x:x[1],reverse=True)[0:10]:
        out += "<li>"+item[0]+"</li>\n"
    out += "</ol>\n"
out += "</html>"

with open('output.html','w') as output:
    output.write(out)
        

    