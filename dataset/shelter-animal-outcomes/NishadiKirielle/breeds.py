import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
matplotlib.rcParams.update({'font.size': 12})

df = pd.read_csv('../input/train.csv', sep=',')

feature = 'Breed'

feature_values_dog = df.loc[df['AnimalType'] == 'Dog',feature]
outcome_dog = df.loc[df['AnimalType'] == 'Dog','OutcomeType']
outcome_dog = np.array(outcome_dog)

# unique outcomes:
unique_outcomes = np.unique(outcome_dog)

breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']

breeds_group = np.array([breeds,groups]).T
dog_groups = np.unique(breeds_group[:,1])
group_values_dog = []

count = 0

not_found = []

for i in feature_values_dog:
    i = i.replace(' Shorthair','')
    i = i.replace(' Longhair','')
    i = i.replace(' Wirehair','')
    i = i.replace(' Rough','')
    i = i.replace(' Smooth Coat','')
    i = i.replace(' Smooth','')
    i = i.replace(' Black/Tan','')
    i = i.replace('Black/Tan ','')
    i = i.replace(' Flat Coat','')
    i = i.replace('Flat Coat ','')
    i = i.replace(' Coat','')
    
    groups = []
    if '/' in i:
        split_i = i.split('/')
        for j in split_i:
            if j[-3:] == 'Mix':
                breed = j[:-4]               
                if breed in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == breed)[0]
                    groups.append(breeds_group[indx,1][0])
                    groups.append('Mix')
                elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                    groups.append('Mix')  
                elif breed == 'Pit Bull':
                    groupd.append('Pit Bull')
                    groups.append('Mix')  
                elif 'Shepherd' in breed:
                    groups.append('Herding')
                    groups.append('Mix')  
                else:
                    not_found.append(breed)
                    groups.append('Unknown')
                    groups.append('Mix')
            else:
                if j in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == j)[0]
                    groups.append(breeds_group[indx,1][0])
                elif np.any([s.lower() in j.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                elif j == 'Pit Bull':
                    groups.append('Pit Bull')
                elif 'Shepherd' in j:
                    groups.append('Herding')
                    groups.append('Mix')  
                else:
                    not_found.append(j)
                    groups.append('Unknown')
    else:

        if i[-3:] == 'Mix':
            breed = i[:-4]
            if breed in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == breed)[0]
                groups.append(breeds_group[indx,1][0])
                groups.append('Mix')
            elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                groups.append(find_group[find_group != 'Unknown'])
                groups.append('Mix') 
            elif breed == 'Pit Bull':
                groups.append('Pit Bull')
                groups.append('Mix') 
            elif 'Shepherd' in breed:
                groups.append('Herding')
                groups.append('Mix')  
            else:
                groups.append('Unknown')
                groups.append('Mix') 
                not_found.append(breed)

        else:
            if i in breeds_group[:,0]:
                indx = np.where(breeds_group[:,0] == i)[0]
                groups.append(breeds_group[indx,1][0])
            elif np.any([s.lower() in i.lower() for s in dog_groups]):
                find_group = [s if s.lower() in i.lower() else 'Unknown' for s in dog_groups]                    
                groups.append(find_group[find_group != 'Unknown'])
            elif i == 'Pit Bull':
                groups.append('Pit Bull')
            elif 'Shepherd' in i:
                groups.append('Herding')
                groups.append('Mix') 
            else:
                groups.append('Unknown') 
                not_found.append(i)
    group_values_dog.append(list(set(groups)))

not_f_unique,counts = np.unique(not_found,return_counts=True)

unique_groups, counts = np.unique(group_values_dog,return_counts=True)

# add mix, pit bull, and unknown to the groups
groups = np.unique(np.append(dog_groups,['Mix','Pit Bull','Unknown']))

print(groups)