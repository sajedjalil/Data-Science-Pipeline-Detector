# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

# Load the train and test datasets to create two DataFrames
train_url = "../input/train.csv"
train = pd.read_csv(train_url)

test_url = "../input/test.csv"
test = pd.read_csv(test_url)


breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']
# print (type(breeds))
# print (breeds[:3])
# print (groups[:3])
breeds_group = np.array([breeds,groups]).T
#
# print (breeds_group[:2])
dog_groups = np.unique(breeds_group[:,1])
# print ("dog_groups", dog_groups[:2])


def get_breed(i):
    if i ==  'Blue Lacy' : return 0
    elif i ==  'Queensland Heeler' : return 0
    elif i ==  'Rhod Ridgeback' : return 1
    elif i ==  'Retriever' : return 2
    elif i ==  'Chinese Sharpei' : return 3
    elif i ==  'Black Mouth Cur' : return 0
    elif i ==  'Catahoula' : return 0
    elif i ==  'Staffordshire' : return 4
    elif i ==  'Affenpinscher' : return 5
    elif i ==  'Afghan Hound' : return 1
    elif i ==  'Airedale Terrier' : return 4
    elif i ==  'Akita' : return 6
    elif i ==  'Australian Kelpie' : return 6
    elif i ==  'Alaskan Malamute' : return 6
    elif i ==  'English Bulldog' : return 3
    elif i ==  'American Bulldog' : return 3
    elif i ==  'American English Coonhound' : return 1
    elif i ==  'American Eskimo Dog (Miniature)' : return 3
    elif i ==  'American Eskimo Dog (Standard)' : return 3
    elif i ==  'American Eskimo Dog (Toy)' : return 5
    elif i ==  'American Foxhound' : return 1
    elif i ==  'American Hairless Terrier' : return 4
    elif i ==  'American Staffordshire Terrier' : return 4
    elif i ==  'American Water Spaniel' : return 2
    elif i ==  'Anatolian Shepherd Dog' : return 6
    elif i ==  'Australian Cattle Dog' : return 0
    elif i ==  'Australian Shepherd' : return 0
    elif i ==  'Australian Terrier' : return 4
    elif i ==  'Basenji' : return 1
    elif i ==  'Basset Hound' : return 1
    elif i ==  'Beagle' : return 1
    elif i ==  'Bearded Collie' : return 0
    elif i ==  'Beauceron' : return 0
    elif i ==  'Bedlington Terrier' : return 4
    elif i ==  'Belgian Malinois' : return 0
    elif i ==  'Belgian Sheepdog' : return 0
    elif i ==  'Belgian Tervuren' : return 0
    elif i ==  'Bergamasco' : return 0
    elif i ==  'Berger Picard' : return 0
    elif i ==  'Bernese Mountain Dog' : return 6
    elif i ==  'Bichon Fris_' : return 3
    elif i ==  'Black and Tan Coonhound' : return 1
    elif i ==  'Black Russian Terrier' : return 6
    elif i ==  'Bloodhound' : return 1
    elif i ==  'Bluetick Coonhound' : return 1
    elif i ==  'Boerboel' : return 6
    elif i ==  'Border Collie' : return 0
    elif i ==  'Border Terrier' : return 4
    elif i ==  'Borzoi' : return 1
    elif i ==  'Boston Terrier' : return 3
    elif i ==  'Bouvier des Flandres' : return 0
    elif i ==  'Boxer' : return 6
    elif i ==  'Boykin Spaniel' : return 2
    elif i ==  'Briard' : return 0
    elif i ==  'Brittany' : return 2
    elif i ==  'Brussels Griffon' : return 5
    elif i ==  'Bull Terrier' : return 4
    elif i ==  'Bull Terrier (Miniature)' : return 4
    elif i ==  'Bulldog' : return 3
    elif i ==  'Bullmastiff' : return 6
    elif i ==  'Cairn Terrier' : return 4
    elif i ==  'Canaan Dog' : return 6
    elif i ==  'Cane Corso' : return 6
    elif i ==  'Cardigan Welsh Corgi' : return 0
    elif i ==  'Cavalier King Charles Spaniel' : return 5
    elif i ==  'Cesky Terrier' : return 4
    elif i ==  'Chesapeake Bay Retriever' : return 2
    elif i ==  'Chihuahua' : return 5
    elif i ==  'Chinese Crested Dog' : return 5
    elif i ==  'Chinese Shar Pei' : return 3
    elif i ==  'Chinook' : return 6
    elif i ==  'Chow Chow' : return 3
    elif i ==  "Cirneco dell'Etna" : return 1
    elif i ==  'Clumber Spaniel' : return 2
    elif i ==  'Cocker Spaniel' : return 2
    elif i ==  'Collie' : return 0
    elif i ==  'Coton de Tulear' : return 3
    elif i ==  'Curly-Coated Retriever' : return 2
    elif i ==  'Dachshund' : return 1
    elif i ==  'Dalmatian' : return 3
    elif i ==  'Dandie Dinmont Terrier' : return 4
    elif i ==  'Doberman Pinsch' : return 6
    elif i ==  'Doberman Pinscher' : return 6
    elif i ==  'Dogue De Bordeaux' : return 6
    elif i ==  'English Cocker Spaniel' : return 2
    elif i ==  'English Foxhound' : return 1
    elif i ==  'English Setter' : return 2
    elif i ==  'English Springer Spaniel' : return 2
    elif i ==  'English Toy Spaniel' : return 5
    elif i ==  'Entlebucher Mountain Dog' : return 0
    elif i ==  'Field Spaniel' : return 2
    elif i ==  'Finnish Lapphund' : return 0
    elif i ==  'Finnish Spitz' : return 3
    elif i ==  'Flat-Coated Retriever' : return 2
    elif i ==  'French Bulldog' : return 3
    elif i ==  'German Pinscher' : return 6
    elif i ==  'German Shepherd' : return 0
    elif i ==  'German Shorthaired Pointer' : return 2
    elif i ==  'German Wirehaired Pointer' : return 2
    elif i ==  'Giant Schnauzer' : return 6
    elif i ==  'Glen of Imaal Terrier' : return 4
    elif i ==  'Golden Retriever' : return 2
    elif i ==  'Gordon Setter' : return 2
    elif i ==  'Great Dane' : return 6
    elif i ==  'Great Pyrenees' : return 6
    elif i ==  'Greater Swiss Mountain Dog' : return 6
    elif i ==  'Greyhound' : return 1
    elif i ==  'Harrier' : return 1
    elif i ==  'Havanese' : return 5
    elif i ==  'Ibizan Hound' : return 1
    elif i ==  'Icelandic Sheepdog' : return 0
    elif i ==  'Irish Red and White Setter' : return 2
    elif i ==  'Irish Setter' : return 2
    elif i ==  'Irish Terrier' : return 4
    elif i ==  'Irish Water Spaniel' : return 2
    elif i ==  'Irish Wolfhound' : return 1
    elif i ==  'Italian Greyhound' : return 5
    elif i ==  'Japanese Chin' : return 5
    elif i ==  'Keeshond' : return 3
    elif i ==  'Kerry Blue Terrier' : return 4
    elif i ==  'Komondor' : return 6
    elif i ==  'Kuvasz' : return 6
    elif i ==  'Labrador Retriever' : return 2
    elif i ==  'Lagotto Romagnolo' : return 2
    elif i ==  'Lakeland Terrier' : return 4
    elif i ==  'Leonberger' : return 6
    elif i ==  'Lhasa Apso' : return 3
    elif i ==  'L_wchen' : return 3
    elif i ==  'Maltese' : return 5
    elif i ==  'Manchester Terrier' : return 4
    elif i ==  'Mastiff' : return 6
    elif i ==  'Miniature American Shepherd' : return 0
    elif i ==  'Miniature Bull Terrier' : return 4
    elif i ==  'Miniature Pinscher' : return 5
    elif i ==  'Miniature Schnauzer' : return 4
    elif i ==  'Neapolitan Mastiff' : return 6
    elif i ==  'Newfoundland' : return 6
    elif i ==  'Norfolk Terrier' : return 4
    elif i ==  'Norwegian Buhund' : return 0
    elif i ==  'Norwegian Elkhound' : return 1
    elif i ==  'Norwegian Lundehund' : return 3
    elif i ==  'Norwich Terrier' : return 4
    elif i ==  'Nova Scotia Duck Tolling Retriever' : return 2
    elif i ==  'Old English Sheepdog' : return 0
    elif i ==  'Otterhound' : return 1
    elif i ==  'Papillon' : return 5
    elif i ==  'Parson Russell Terrier' : return 4
    elif i ==  'Pekingese' : return 5
    elif i ==  'Pembroke Welsh Corgi' : return 0
    elif i ==  'Petit Basset Griffon Vend_en' : return 1
    elif i ==  'Pharaoh Hound' : return 1
    elif i ==  'Plott' : return 1
    elif i ==  'Pointer' : return 2
    elif i ==  'Polish Lowland Sheepdog' : return 0
    elif i ==  'Pomeranian' : return 5
    elif i ==  'Standard Poodle' : return 3
    elif i ==  'Miniature Poodle' : return 3
    elif i ==  'Toy Poodle' : return 5
    elif i ==  'Portuguese Podengo Pequeno' : return 1
    elif i ==  'Portuguese Water Dog' : return 6
    elif i ==  'Pug' : return 5
    elif i ==  'Puli' : return 0
    elif i ==  'Pyrenean Shepherd' : return 0
    elif i ==  'Rat Terrier' : return 4
    elif i ==  'Redbone Coonhound' : return 1
    elif i ==  'Rhodesian Ridgeback' : return 1
    elif i ==  'Rottweiler' : return 6
    elif i ==  'Russell Terrier' : return 4
    elif i ==  'St. Bernard' : return 6
    elif i ==  'Saluki' : return 1
    elif i ==  'Samoyed' : return 6
    elif i ==  'Schipperke' : return 3
    elif i ==  'Scottish Deerhound' : return 1
    elif i ==  'Scottish Terrier' : return 4
    elif i ==  'Sealyham Terrier' : return 4
    elif i ==  'Shetland Sheepdog' : return 0
    elif i ==  'Shiba Inu' : return 3
    elif i ==  'Shih Tzu' : return 5
    elif i ==  'Siberian Husky' : return 6
    elif i ==  'Silky Terrier' : return 5
    elif i ==  'Skye Terrier' : return 4
    elif i ==  'Sloughi' : return 1
    elif i ==  'Smooth Fox Terrier' : return 4
    elif i ==  'Soft-Coated Wheaten Terrier' : return 4
    elif i ==  'Spanish Water Dog' : return 0
    elif i ==  'Spinone Italiano' : return 2
    elif i ==  'Staffordshire Bull Terrier' : return 4
    elif i ==  'Standard Schnauzer' : return 6
    elif i ==  'Sussex Spaniel' : return 2
    elif i ==  'Swedish Vallhund' : return 0
    elif i ==  'Tibetan Mastiff' : return 6
    elif i ==  'Tibetan Spaniel' : return 3
    elif i ==  'Tibetan Terrier' : return 3
    elif i ==  'Toy Fox Terrier' : return 5
    elif i ==  'Treeing Walker Coonhound' : return 1
    elif i ==  'Vizsla' : return 2
    elif i ==  'Weimaraner' : return 2
    elif i ==  'Welsh Springer Spaniel' : return 2
    elif i ==  'Welsh Terrier' : return 4
    elif i ==  'West Highland White Terrier' : return 4
    elif i ==  'Whippet' : return 1
    elif i ==  'Wire Fox Terrier' : return 4
    elif i ==  'Wirehaired Pointing Griffon' : return 2
    elif i ==  'Wirehaired Vizsla' : return 2
    elif i ==  'Xoloitzcuintli' : return 3
    elif i ==  'Yorkshire Terrier' : return 5
    elif i == 'Burmese':
        return 21
    elif i == 'Sphynx':
        return 22
    elif i == 'Devon Rex':
        return 7
    elif i == 'Rex':
        return 8
    elif i == 'Bengal':
        return 9
    elif i == 'British':
        return 10
    elif i == 'Japanese Bobtail':
        return 11
    elif i == 'Ragdoll':
        return 12
    elif i == 'Persian':
        return 13
    elif i == 'Himalayan':
        return 14
    elif i == 'Russian Blue':
        return 15
    elif i == 'Maine Coon':
        return 16
    elif i == 'Manx':
        return 17
    elif i == 'Snowshoe':
        return 18
    elif i == 'Siamese':
        return 19
    elif i == 'Domestic':
        return 20
    return 25

    

# def get_breed(i):

 

#     i = i.replace(' Shorthair','')
#     i = i.replace(' Longhair','')
#     i = i.replace(' Wirehair','')
#     i = i.replace(' Rough','')
#     i = i.replace(' Smooth Coat','')
#     i = i.replace(' Smooth','')
#     i = i.replace(' Black/Tan','')
#     i = i.replace('Black/Tan ','')
#     i = i.replace(' Flat Coat','')
#     i = i.replace('Flat Coat ','')
#     i = i.replace(' Coat','')

#     if '/' in i:
#         split_i = i.split('/')
#         for j in split_i:
#             if j[-3:] == 'Mix':
#                 breed = j[:-4]
#                 if breed in breeds_group[:,0]:
#                     # indx = np.where(breeds_group[:,0] == breed)[0]
#                     # groups.append(breeds_group[indx,1][0])
#                     # groups.append('Mix')
#                     return 3
#                 elif np.any([s.lower() in breed.lower() for s in dog_groups]):
#                     # find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]
#                     # groups.append(find_group[find_group != 'Unknown'])
#                     # groups.append('Mix')
#                     return 0
#                 elif breed == 'Pit Bull':
#                     # groupd.append('Pit Bull')
#                     # groups.append('Mix')
#                     return 1
#                 elif 'Shepherd' in breed:
#                     # groups.append('Herding')
#                     # groups.append('Mix')
#                     return 2
#                 else:
#                     # not_found.append(breed)
#                     # groups.append('Unknown')
#                     # groups.append('Mix')
#                     return 0
#             else:
#                 if j in breeds_group[:,0]:
#                     # indx = np.where(breeds_group[:,0] == j)[0]
#                     # groups.append(breeds_group[indx,1][0])
#                     return 3
#                 elif np.any([s.lower() in j.lower() for s in dog_groups]):
#                     # find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]
#                     # groups.append(find_group[find_group != 'Unknown'])
#                     return 0
#                 elif j == 'Pit Bull':
#                     # groups.append('Pit Bull')
#                     return 1
#                 elif 'Shepherd' in j:
#                     # groups.append('Herding')
#                     # groups.append('Mix')
#                     return 2
#                 else:
#                     # not_found.append(j)
#                     # groups.append('Unknown')
#                     return 0
#     else:

#         if i[-3:] == 'Mix':
#             breed = i[:-4]
#             if breed in breeds_group[:, 0]:
#                 # indx = np.where(breeds_group[:, 0] == breed)[0]
#                 # groups.append(breeds_group[indx, 1][0])
#                 # groups.append('Mix')
#                 return 3
#             elif np.any([s.lower() in breed.lower() for s in dog_groups]):
#                 # find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]
#                 # groups.append(find_group[find_group != 'Unknown'])
#                 # groups.append('Mix')
#                 return 0
#             elif breed == 'Pit Bull':
#                 # groups.append('Pit Bull')
#                 # groups.append('Mix')
#                 return 1
#             elif 'Shepherd' in breed:
#                 # groups.append('Herding')
#                 # groups.append('Mix')
#                 return 2
#             else:
#                 # groups.append('Unknown')
#                 # groups.append('Mix')
#                 # not_found.append(breed)
#                 return 0

#         else:
#             if i in breeds_group[:, 0]:
#                 # indx = np.where(breeds_group[:, 0] == i)[0]
#                 # groups.append(breeds_group[indx, 1][0])
#                 return 3
#             elif np.any([s.lower() in i.lower() for s in dog_groups]):
#                 # find_group = [s if s.lower() in i.lower() else 'Unknown' for s in dog_groups]
#                 # groups.append(find_group[find_group != 'Unknown'])
#                 return 0
#             elif i == 'Pit Bull':
#                 # groups.append('Pit Bull')
#                 return 1
#             elif 'Shepherd' in i:
#                 # groups.append('Herding')
#                 # groups.append('Mix')
#                 return 2
#             else:
#                 # groups.append('Unknown')
#                 # not_found.append(i)
#                 return 0
#     return 4

def get_animal_type(x):
    x = str(x)
    if x.find('Dog') >= 0: return 0
    if x.find('Cat') >= 0: return 1
    return 2


def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 0
    if x.find('Female') >= 0: return 1
    return 2


def get_neutered(x):
    x = str(x)
    if x.find('Neutered Male') >= 0: return 0
    if x.find('Spayed Female') >= 0: return 1
    if x.find('Intact Male') >= 0: return 2
    if x.find('Intact Female') >= 0: return 3
    return 4


# def get_neutered(x):
#     x = str(x)
#     if x.find('Neutered Male') >= 0: return 0
#     if x.find('Spayed Female') >= 0: return 0
#     if x.find('Intact Male') >= 0: return 1
#     if x.find('Intact Female') >= 0: return 2
#     return 3

def get_hasName(x):
    x = str(x)
    if len(x) >= 0: return 1
    return 0


def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 1
    return 0


def get_simple_color(x):
    x = str(x)
    if x.find('/') >= 0: return 1
    return 0


def calc_age_in_days(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age * 365
    if x.find('month') > -1: return age * 30
    if x.find('week') > -1: return age * 7
    if x.find('day') > -1:
        return age
    else:
        return 0


def calc_age_in_years(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age
    if x.find('month') > -1: return age / 12.
    if x.find('week') > -1: return age / 52.
    if x.find('day') > -1:
        return age / 365.
    else:
        return 0


def get_hour(x):
    x = str(x)
    split_x = x.split()
    hour = int(x[1].split(':')[0])
    if hour > 2 & hour < 5: return 0
    if hour > 4 & hour < 7: return 1
    if hour > 6 & hour < 9: return 2
    if hour > 8 & hour < 11: return 3
    if hour > 10 & hour < 13: return 4
    if hour > 12 & hour < 15: return 5
    if hour > 14 & hour < 17: return 6
    if hour > 16 & hour < 19: return 7
    if hour > 18 & hour < 21: return 8
    if hour > 20 & hour < 23: return 9
    if hour > 22 & hour < 24:
        return 10
    else:
        return 11


def get_month(x):
    x = str(x)
    split_x = x.split()
    month = int(x[0].split('/')[0])
    return month


train['AnimalType'] = train.AnimalType.apply(get_animal_type)
train['Sex'] = train.SexuponOutcome.apply(get_sex)
train['Neutered'] = train.SexuponOutcome.apply(get_neutered)
train['Name'] = train.Name.apply(get_hasName)
train['Mix'] = train.Breed.apply(get_mix)
train['AgeInYears'] = train.AgeuponOutcome.apply(calc_age_in_years)
train['AgeInDays'] = train.AgeuponOutcome.apply(calc_age_in_days)
train['Month'] = train.DateTime.apply(get_month)
train['Color'] = train.Color.apply(get_simple_color)
train['Breed'] = train.Breed.apply(get_breed)

target = train["OutcomeType"].values
features_one = train[["AnimalType", "Sex", "AgeInDays", 'Neutered','Breed']].values

#  # Fit your first decision tree: my_tree_one
# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)

# Building and fitting my_forest
forest = RandomForestClassifier(n_estimators=1000, random_state=1)
my_forest = forest.fit(features_one, target)

# tree.export_graphviz(my_tree_one, out_file='tree.dot')

# Look at the importance and score of the included features
# print("Feature Importance")
# print(my_tree_one.feature_importances_)
# print(my_tree_one.score(features_one, target))
print(my_forest.score(features_one, target))

test['AnimalType'] = test.AnimalType.apply(get_animal_type)
test['Sex'] = test.SexuponOutcome.apply(get_sex)
test['AgeInYears'] = test.AgeuponOutcome.apply(calc_age_in_years)
test['AgeInDays'] = test.AgeuponOutcome.apply(calc_age_in_days)
test['Neutered'] = test.SexuponOutcome.apply(get_neutered)
test['Name'] = test.Name.apply(get_hasName)
test['Mix'] = test.Breed.apply(get_mix)
test['Month'] = test.DateTime.apply(get_month)
test['Color'] = test.Color.apply(get_simple_color)
test['Breed'] = test.Breed.apply(get_breed)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
#test_features = test[["AnimalType", "Sex", "AgeInDays", 'Neutered','Mix','Month','Color','Name']].values
test_features = test[["AnimalType", "Sex", "AgeInDays", 'Neutered','Breed']].values

# pred_forest = my_forest.predict(___)
my_prediction = my_forest.predict(test_features)
ID = np.array(test["ID"]).astype(int)
my_solution = pd.DataFrame(my_prediction, ID, columns=["OutcomeType"])

# Write your solution to a csv file with the name my_solution.csv
my_solution["Adoption"] = 0
my_solution["Adoption"][my_solution["OutcomeType"] == "Adoption"] = 1
my_solution["Died"] = 0
my_solution["Died"][my_solution["OutcomeType"] == "Died"] = 1
my_solution["Euthanasia"] = 0
my_solution["Euthanasia"][my_solution["OutcomeType"] == "Euthanasia"] = 1
my_solution["Return_to_owner"] = 0
my_solution["Return_to_owner"][my_solution["OutcomeType"] == "Return_to_owner"] = 1
my_solution["Transfer"] = 0
my_solution["Transfer"][my_solution["OutcomeType"] == "Transfer"] = 1
my_solution = my_solution.drop('OutcomeType', 1)

my_solution.to_csv("Random_Forest_BreedGroups.csv", index_label=["ID"])
