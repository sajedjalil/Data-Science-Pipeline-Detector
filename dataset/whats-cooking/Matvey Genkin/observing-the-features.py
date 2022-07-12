



def main():
    pass

if __name__ == '__main__': 
    main()

import json

with open('../input/train.json') as data_file:
    train_data_json = json.load(data_file)

with open('../input/test.json') as data_file:
    test_data_json=json.load(data_file)

print(type(train_data_json))
