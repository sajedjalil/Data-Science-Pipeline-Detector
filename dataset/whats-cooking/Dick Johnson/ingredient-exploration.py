import json
import string

with open('../input/train.json', 'r') as fh:
    train = json.load(fh)
    
ingredients = set()
for recipe in train:
    for ingred in recipe['ingredients']:
        ingredients.add(ingred.lower())
        
# Print multi-word ingredients
multiword = set()
for i in ingredients:
    if len(i.split()) != 1:
        multiword.add(i)
        
# Print Long Ingredients:
longingredients = set()
for i in ingredients:
    if len(i.split()) == 1 and len(i) > 10:
        longingredients.add(i)
        
# Contains special characters
specialchars = set()
for i in ingredients:
    for letter in i:
        if letter in string.punctuation:
            specialchars.add(i)
            
#Brands? Unicode shenanigans?
brands = set()
for i in ingredients:
    if u'\u00AE' in i or u'\u2122' in i:
        brands.add(i)

print('Total ingredients: %d' % len(ingredients))
print('Length of brands: %d' % len(brands))
print('Length of longingredients: %d' % len(longingredients))
print('Length of multiword: %d' % len(multiword))
print('Length of specialchars: %d' % len(specialchars))

print(list(brands)[:3])
print(list(longingredients)[:3])
print(list(multiword)[:3])
print(list(specialchars)[:3])