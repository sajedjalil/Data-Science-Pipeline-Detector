import json

  
with open('../input/train.json') as json_file:
    json_data = json.load(json_file)
    print(json_data[0]['id'])
    print(json_data[0]['cuisine'])
    print(json_data[0]['ingredients'])

