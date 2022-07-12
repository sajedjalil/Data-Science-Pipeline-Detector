import json
with open('../input/train.json') as data_file:    
  train = json.load(data_file)

cousine_ingredients = {}
cousine_totals = {}

for recipe in train:
    ingredients = recipe['ingredients']
    cousine_name = recipe['cuisine']
    cousine_map = cousine_ingredients.get(cousine_name, {})
    cousine_total_icount = cousine_totals.get(cousine_name, 0)
    for iname in ingredients:
        iname = iname.lower()
        icount = cousine_map.get(iname, 0)
        cousine_map[iname] = icount+1
        cousine_total_icount = cousine_total_icount + 1
    cousine_ingredients[cousine_name] = cousine_map
    cousine_totals[cousine_name] = cousine_total_icount

with open('../input/test.json') as test_data_file:    
    test_data = json.load(test_data_file)

recipe_map = {}
for recipe in test_data:
    recipe_id = recipe['id']
    recipe_ingredients = recipe['ingredients']
    recipe_cousine_map = {}
    for cousine_name in cousine_ingredients:
        cousine_map = cousine_ingredients[cousine_name]
        cousine_score = 0
        for iname in recipe_ingredients:
            ingredient_score = cousine_map.get(iname,0)
            cousine_score = cousine_score + ingredient_score            
        cousine_score_normalized = cousine_score/cousine_totals[cousine_name]
        recipe_cousine_map[cousine_name] = cousine_score_normalized
    recipe_map[recipe_id] = recipe_cousine_map   
    
recipe_results = {}
for recipe_id in recipe_map:
    recipe_cousine_map = recipe_map[recipe_id]
    # get cousine with max score from recipe_cousine_map
    cousine_name = max(recipe_cousine_map, key=recipe_cousine_map.get)
    recipe_results[recipe_id]=cousine_name
    
with open('./result.csv', 'w') as rst_file:
    rst_file.write('id,cuisine'+'\n')
    for recipe_id in recipe_results:
        cousine_name = recipe_results[recipe_id]
        rst_file.write(str(recipe_id)+','+cousine_name+'\n')
