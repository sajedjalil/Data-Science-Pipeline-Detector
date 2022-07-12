import pandas as pd
import xgboost as xgb


# Initial data pre processing
def data_pre_process_initial(train, test):
    train.drop('AnimalID', axis=1, inplace=True)

    ## Outcome Subtype

    # Initializing Outcome Subtypes
    train['Aggressive'] = 0
    train['At Vet'] = 0
    train['Barn'] = 0
    train['Behavior'] = 0
    train['Court/Investigation'] = 0
    train['Enroute'] = 0
    train['Foster'] = 0
    train['In Foster'] = 0
    train['In Kennel'] = 0
    train['In Surgery'] = 0
    train['Medical'] = 0
    train['Offsite'] = 0
    train['Partner'] = 0
    train['Rabies Risk'] = 0
    train['SCRP'] = 0
    train['Suffering'] = 0

    # Updating Outcome Subtypes
    train.ix[train.OutcomeSubtype == 'Aggressive', 'Aggressive'] = 1
    train.ix[train.OutcomeSubtype == 'At Vet', 'At Vet'] = 1
    train.ix[train.OutcomeSubtype == 'Barn', 'Barn'] = 1
    train.ix[train.OutcomeSubtype == 'Behavior', 'Behavior'] = 1
    train.ix[train.OutcomeSubtype == 'Court/Investigation', 'Court/Investigation'] = 1
    train.ix[train.OutcomeSubtype == 'Enroute', 'Enroute'] = 1
    train.ix[train.OutcomeSubtype == 'Foster', 'Foster'] = 1
    train.ix[train.OutcomeSubtype == 'In Foster', 'In Foster'] = 1
    train.ix[train.OutcomeSubtype == 'In Kennel', 'In Kennel'] = 1
    train.ix[train.OutcomeSubtype == 'In Surgery', 'In Surgery'] = 1
    train.ix[train.OutcomeSubtype == 'Medical', 'Medical'] = 1
    train.ix[train.OutcomeSubtype == 'Offsite', 'Offsite'] = 1
    train.ix[train.OutcomeSubtype == 'Partner', 'Partner'] = 1
    train.ix[train.OutcomeSubtype == 'Rabies Risk', 'Rabies Risk'] = 1
    train.ix[train.OutcomeSubtype == 'SCRP', 'SCRP'] = 1
    train.ix[train.OutcomeSubtype == 'Suffering', 'Suffering'] = 1

    # Name availability
    train['Name_bool'] = pd.isnull(train['Name']).astype(int)
    test['Name_bool'] = pd.isnull(test['Name']).astype(int)

    # OutcomeType
    train.ix[train.OutcomeType == 'Return_to_owner', 'Target'] = 0
    train.ix[train.OutcomeType == 'Euthanasia', 'Target'] = 1
    train.ix[train.OutcomeType == 'Adoption', 'Target'] = 2
    train.ix[train.OutcomeType == 'Transfer', 'Target'] = 3
    train.ix[train.OutcomeType == 'Died', 'Target'] = 4

    # Animal Type Cat (Binary value)
    train.ix[train.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    train.ix[train.AnimalType == 'Cat', 'AnimalType_cat'] = 1

    test.ix[test.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    test.ix[test.AnimalType == 'Cat', 'AnimalType_cat'] = 1

    ## SexuponOutcome

    train.ix[train.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    train.ix[train.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    train.ix[train.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    train.ix[train.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    train.ix[train.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4

    test.ix[test.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    test.ix[test.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    test.ix[test.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    test.ix[test.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    test.ix[test.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4

    # Male / Female
    train.ix[train.SexuponOutcome == 'Intact Female', 'Type_sex'] = 1
    train.ix[train.SexuponOutcome == 'Spayed Female', 'Type_sex'] = 1
    train.ix[train.SexuponOutcome == 'Neutered Male', 'Type_sex'] = 2
    train.ix[train.SexuponOutcome == 'Intact Male', 'Type_sex'] = 2
    train.ix[train.SexuponOutcome == 'Unknown', 'Type_sex'] = 3

    test.ix[test.SexuponOutcome == 'Intact Female', 'Type_sex'] = 1
    test.ix[test.SexuponOutcome == 'Spayed Female', 'Type_sex'] = 1
    test.ix[test.SexuponOutcome == 'Neutered Male', 'Type_sex'] = 2
    test.ix[test.SexuponOutcome == 'Intact Male', 'Type_sex'] = 2
    test.ix[test.SexuponOutcome == 'Unknown', 'Type_sex'] = 3

    # Sprayed / Intact
    train.ix[train.SexuponOutcome == 'Intact Female', 'sprayed'] = 1
    train.ix[train.SexuponOutcome == 'Spayed Female', 'sprayed'] = 2
    train.ix[train.SexuponOutcome == 'Neutered Male', 'sprayed'] = 2
    train.ix[train.SexuponOutcome == 'Intact Male', 'sprayed'] = 1
    train.ix[train.SexuponOutcome == 'Unknown', 'sprayed'] = 3

    test.ix[test.SexuponOutcome == 'Intact Female', 'sprayed'] = 1
    test.ix[test.SexuponOutcome == 'Spayed Female', 'sprayed'] = 2
    test.ix[test.SexuponOutcome == 'Neutered Male', 'sprayed'] = 2
    test.ix[test.SexuponOutcome == 'Intact Male', 'sprayed'] = 1
    test.ix[test.SexuponOutcome == 'Unknown', 'sprayed'] = 3

    # AgeuponOutcome
    train['AgeuponOutcome_In_Days'] = 0
    train['Age_unit'] = ''

    test['AgeuponOutcome_In_Days'] = 0
    test['Age_unit'] = ''

    for i in range(0, train.shape[0]):
        try:
            train['AgeuponOutcome_In_Days'][i] = train['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            train['AgeuponOutcome_In_Days'][i] = 0
        try:
            train['Age_unit'][i] = train['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            train['Age_unit'][i] = 'day'

    for i in range(0, test.shape[0]):
        try:
            test['AgeuponOutcome_In_Days'][i] = test['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            test['AgeuponOutcome_In_Days'][i] = 0
        try:
            test['Age_unit'][i] = test['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            test['Age_unit'][i] = 'day'

    train.replace('year', 'years', inplace=True)
    train.replace('day', 'days', inplace=True)
    train.replace('month', 'months', inplace=True)
    train.replace('week', 'weeks', inplace=True)
    test.replace('year', 'years', inplace=True)
    test.replace('day', 'days', inplace=True)
    test.replace('month', 'months', inplace=True)
    test.replace('week', 'weeks', inplace=True)

    for i in range(0, train.shape[0]):
        if train['Age_unit'][i] == 'years':
            train['AgeuponOutcome_In_Days'][i] = train['AgeuponOutcome_In_Days'][i] * 365
        if train['Age_unit'][i] == 'months':
            train['AgeuponOutcome_In_Days'][i] = train['AgeuponOutcome_In_Days'][i] * 30
        if train['Age_unit'][i] == 'weeks':
            train['AgeuponOutcome_In_Days'][i] = train['AgeuponOutcome_In_Days'][i] * 7
    for i in range(0, test.shape[0]):
        if test['Age_unit'][i] == 'years':
            test['AgeuponOutcome_In_Days'][i] = test['AgeuponOutcome_In_Days'][i] * 365
        if test['Age_unit'][i] == 'months':
            test['AgeuponOutcome_In_Days'][i] = test['AgeuponOutcome_In_Days'][i] * 30
        if test['Age_unit'][i] == 'weeks':
            test['AgeuponOutcome_In_Days'][i] = test['AgeuponOutcome_In_Days'][i] * 7

    train['AgeuponOutcome_cat'] = 2
    test['AgeuponOutcome_cat'] = 2

    for i in range(0, train.shape[0]):
        if train['AgeuponOutcome_In_Days'][i] <= 480:
            train['AgeuponOutcome_cat'][i] = 1
        if 480 < train['AgeuponOutcome_In_Days'][i] <= 2555:
            train['AgeuponOutcome_cat'][i] = 2
        if 2555 < train['AgeuponOutcome_In_Days'][i]:
            train['AgeuponOutcome_cat'][i] = 3
    for i in range(0, test.shape[0]):
        if test['AgeuponOutcome_In_Days'][i] <= 480:
            test['AgeuponOutcome_cat'][i] = 1
        if 480 < test['AgeuponOutcome_In_Days'][i] < 2555:
            test['AgeuponOutcome_cat'][i] = 2
        if 2555 < test['AgeuponOutcome_In_Days'][i]:
            test['AgeuponOutcome_cat'][i] = 3

    # Breed split
    train['Breed1'] = ''
    train['Breed2'] = ''
    train['Breed11'] = ''

    test['Breed1'] = ''
    test['Breed2'] = ''
    test['Breed11'] = ''

    for i in range(0, test.shape[0]):
        try:
            test['Breed1'][i] = test['Breed'][i].split('/')[0]
        except IndexError:
            test['Breed1'][i] = test['Breed'][i]
        try:
            test['Breed2'][i] = test['Breed'][i].split('/')[1]
        except IndexError:
            test['Breed2'][i] = 'One_Breed'
    for i in range(0, train.shape[0]):
        try:
            train['Breed1'][i] = train['Breed'][i].split('/')[0]
        except IndexError:
            train['Breed1'][i] = train['Breed'][i]
        try:
            train['Breed2'][i] = train['Breed'][i].split('/')[1]
        except IndexError:
            train['Breed2'][i] = 'One_Breed'

    for i in range(0, test.shape[0]):
        try:
            test['Breed11'][i] = test['Breed1'][i].split(' Mix')[0]
        except IndexError:
            test['Breed11'][i] = test['Breed1'][i]
    for i in range(0, train.shape[0]):
        try:
            train['Breed11'][i] = train['Breed1'][i].split(' Mix')[0]
        except IndexError:
            train['Breed11'][i] = train['Breed1'][i]

    train['Breed11'] = train['Breed11'].str.upper()
    test['Breed11'] = test['Breed11'].str.upper()

    # Mix / Not Mix
    train['Is_Mixed'] = 0
    test['Is_Mixed'] = 0
    for i in range(0, train.shape[0]):
        if train['Breed'][i].find(' Mix') >= 0:
            train['Is_Mixed'][i] = 1

    for i in range(0, test.shape[0]):
        if test['Breed'][i].find(' Mix') >= 0:
            test['Is_Mixed'][i] = 1

    # Color split
    train['Color1'] = ''
    train['Color2'] = ''
    train['Color_prin1'] = ''
    train['Shade1'] = ''
    train['Color_prin2'] = ''
    train['Shade2'] = ''

    test['Color1'] = ''
    test['Color2'] = ''
    test['Color_prin1'] = ''
    test['Shade1'] = ''
    test['Color_prin2'] = ''
    test['Shade2'] = ''

    for i in range(0, train.shape[0]):
        train['Color1'][i] = train['Color'][i].split('/')[0]
        try:
            train['Color2'][i] = train['Color'][i].split('/')[1]
        except IndexError:
            train['Color2'][i] = ''
        train['Color_prin1'][i] = train['Color1'][i].split(' ')[0]
        try:
            train['Shade1'][i] = train['Color1'][i].split(' ')[1]
        except IndexError:
            train['Shade1'][i] = ''
        train['Color_prin2'][i] = train['Color2'][i].split(' ')[0]
        try:
            train['Shade2'][i] = train['Color2'][i].split(' ')[1]
        except IndexError:
            train['Shade2'][i] = ''

    for i in range(0, test.shape[0]):
        test['Color1'][i] = test['Color'][i].split('/')[0]
        try:
            test['Color2'][i] = test['Color'][i].split('/')[1]
        except IndexError:
            test['Color2'][i] = ''
        test['Color_prin1'][i] = test['Color1'][i].split(' ')[0]
        try:
            test['Shade1'][i] = test['Color1'][i].split(' ')[1]
        except IndexError:
            test['Shade1'][i] = ''
        test['Color_prin2'][i] = test['Color2'][i].split(' ')[0]
        try:
            test['Shade2'][i] = test['Color2'][i].split(' ')[1]
        except IndexError:
            test['Shade2'][i] = ''

    # Datetime
    train['year'] = pd.DatetimeIndex(train['DateTime']).year
    train['month'] = pd.DatetimeIndex(train['DateTime']).month
    train['day'] = pd.DatetimeIndex(train['DateTime']).day
    train['hour'] = pd.DatetimeIndex(train['DateTime']).hour
    train['minute'] = pd.DatetimeIndex(train['DateTime']).minute
    train['quarter'] = pd.DatetimeIndex(train['DateTime']).quarter
    train['weekofyear'] = pd.DatetimeIndex(train['DateTime']).weekofyear

    test['year'] = pd.DatetimeIndex(test['DateTime']).year
    test['month'] = pd.DatetimeIndex(test['DateTime']).month
    test['day'] = pd.DatetimeIndex(test['DateTime']).day
    test['hour'] = pd.DatetimeIndex(test['DateTime']).hour
    test['minute'] = pd.DatetimeIndex(test['DateTime']).minute
    test['quarter'] = pd.DatetimeIndex(test['DateTime']).quarter
    test['weekofyear'] = pd.DatetimeIndex(test['DateTime']).weekofyear

    return train.shape, test.shape

# Breed data pre processing according to breed popularity
def breed_data_pre_process(data_set):
    animal_breeds = ['LABRADOR RETRIEVER', 'GERMAN SHEPHERD', 'GOLDEN RETRIEVER', 'AMERICAN BULLDOG', 'BEAGLE',
                     'FRENCH BULLDOG',
                     'YORKSHIRE TERRIER', 'MINIATURE POODLE', 'ROTTWEILER', 'BOXER', 'German Shorthair Pointer',
                     'SIBERIAN HUSKY',
                     'DACHSHUND', 'Pinscher', 'GREAT DANES', 'MINIATURE SCHNAUZER', 'AUSTRALIAN SHEPHERD',
                     'CAVALIER KING CHARLES SPANIEL', 'SHIH TZU', 'PEMBROKE WELSH CORGI', 'POMERANIAN',
                     'BOSTON TERRIER',
                     'SHETLAND SHEEPDOG', 'HAVANESE', 'MASTIFF', 'BRITTANY', 'ENGLISH SPRINGER SPANIEL', 'CHIHUAHUA',
                     'BERNESE MOUNTAIN DOG', 'COCKER SPANIEL', 'MALTESE', 'VIZSLA', 'PUG', 'WEIMARANER', 'CANE CORSO',
                     'COLLIE',
                     'NEWFOUNDLAND', 'BORDER COLLIE', 'BASSET HOUND', 'RHODESIAN RIDGEBACK',
                     'WEST HIGHLAND WHITE TERRIER',
                     'CHESAPEAKE BAY RETRIEVER', 'BULLMASTIFF', 'BICHON FRISE', 'SHIBA INU', 'AKITA',
                     'SOFT COATED WHEATEN TERRIER', 'PAPILLON', 'BLOODHOUND', 'St. Bernard Smooth Coat',
                     'St. Bernard Rough Coat',
                     'BELGIAN MALINOIS', 'PORTUGUESE WATER DOG', 'AIREDALE TERRIER', 'ALASKAN MALAMUTE', 'BULL TERRIER',
                     'AUSTRALIAN CATTLE DOG', 'WHIPPET', 'SCOTTISH TERRIER', 'CHINESE SHARPEI',
                     'ENGLISH COCKER SPANIEL',
                     'SAMOYED', 'DALMATIAN', 'DOGUE DE BORDEAUX', 'MINIATURE PINSCHER', 'LHASA APSO',
                     'WIREHAIRED POINTING GRIFFON', 'GREAT PYRENEES', 'GERMAN WIREHAIRED POINTER', 'IRISH WOLFHOUND',
                     'CAIRN TERRIER', 'ITALIAN GREYHOUND', 'IRISH SETTER', 'CHOW CHOW', 'OLD ENGLISH SHEEPDOG',
                     'CHINESE CRESTED',
                     'CARDIGAN WELSH CORGI', 'AMERICAN STAFFORDSHIRE TERRIER', 'GREATER SWISS MOUNTAIN DOG',
                     'STAFFORDSHIRE BULL TERRIER', 'PEKINGESE', 'GIANT SCHNAUZER', 'BORDER TERRIER',
                     'BOUVIER DES FLANDRES',
                     'KEESHONDEN', 'COTON DE TULEAR', 'FLAT COAT RETRIEVER', 'BASENJI', 'NORWEGIAN ELKHOUND', 'BORZOI',
                     'TIBETAN TERRIER', 'STANDARD SCHNAUZER', 'ANATOLIAN SHEPHERD DOG', 'LEONBERGER',
                     'WIRE FOX TERRIER',
                     'BRUSSELS GRIFFON', 'ENGLISH SETTER', 'JAPANESE CHIN', 'BELGIAN TERVUREN',
                     'NOVA SCOTIA DUCK TOLLING RETRIEVER', 'AFGHAN HOUND', 'RAT TERRIER', 'SILKY TERRIER',
                     'NORWICH TERRIER',
                     'RUSSELL TERRIER', 'GORDON SETTER', 'NEAPOLITAN MASTIFF', 'BOYKIN SPANIEL', 'WELSH TERRIER',
                     'SCHIPPERKE',
                     'TOY FOX TERRIER', 'PARSON RUSSELL TERRIER', 'SPINONI ITALIANI', 'IRISH TERRIER', 'POINTER',
                     'TIBETAN SPANIEL', 'BLACK RUSSIAN TERRIER', 'TREEING WALKER COONHOUND', 'AMERICAN ESKIMO DOG',
                     'BEARDED COLLIE', 'BELGIAN SHEEPDOG', 'MINIATURE BULL TERRIER', 'SMOOTH FOX TERRIER',
                     'BLUETICK COONHOUND',
                     'KERRY BLUE TERRIER', 'AUSTRALIAN TERRIER', 'BOERBOEL', 'BLACK AND TAN COONHOUND',
                     'WELSH SPRINGER SPANIEL',
                     'ENGLISH TOY SPANIEL', 'BRIARD', 'NORFOLK TERRIER', 'SALUKI', 'TIBETAN MASTIFF', 'CLUMBER SPANIEL',
                     'XOLOITZCUINTLI', 'AFFENPINSCHER', 'MANCHESTER TERRIER', 'GERMAN PINSCHER', 'REDBONE COONHOUND',
                     'ICELANDIC SHEEPDO', 'LAKELAND TERRIER', 'BEAUCERON', 'PETIT BASSET GRIFFON VENDEEN',
                     'IRISH WATER SPANIEL',
                     'FIELD SPANIEL', 'BEDLINGTON TERRIER', 'GREYHOUND', 'IRISH RED AND WHITE SETTER', 'PLOTT',
                     'KUVASZOK',
                     'CURLY COAT RETRIEVER', 'SCOTTISH DEERHOUND', 'PORTUGUESE PODENGO PEQUENO', 'PULIK',
                     'SWEDISH VALLHUND',
                     'WIREHAIRED VIZSLA', 'AMERICAN WATER SPANIEL', 'SEALYHAM TERRIER', 'ENTLEBUCHER MOUNTAIN DOG',
                     'IBIZAN HOUND',
                     'LOWCHEN', 'CIRNECHI DE L ETNA', 'KOMONDOROK', 'POLISH LOWLAND SHEEPDOG', 'NORWEGIAN BUHUND',
                     'AMERICAN ENGLISH COONHOUND', 'SPANISH WATER DOG', 'GLEN OF IMAAL TERRIER', 'FINNISH LAPPHUND',
                     'CANAAN DOG',
                     'PHARAOH HOUND', 'DANDIE DINMONT TERRIER', 'SUSSEX SPANIEL', 'BERGAMASCO', 'SKYE TERRIER',
                     'PYRENEAN SHEPHERD', 'CHINOOK', 'FINNISH SPITZ', 'CESKY TERRIER', 'OTTERHOUND',
                     'AMERICAN FOXHOUND',
                     'NORWEGIAN LUNDEHUND', 'HARRIER', 'ENGLISH FOXHOUND']

    # 2013 Breed Popularity Calculation
    breed_popularity_scores_2013 = ['1', '2', '3', '5', '4', '11', '6', '8', '9', '7', '13', '14', '10', '12', '16',
                                    '17', '20', '18', '15', '24',
                                    '19', '23', '21', '25', '26', '30', '28', '22', '32', '29', '27', '34', '31', '33',
                                    '50', '35', '37', '44',
                                    '41', '39', '36', '43', '40', '40', '46', '45', '51', '38', '48', '47', '47', '60',
                                    '49', '56', '57', '52',
                                    '58', '59', '55', '54', '62', '67', '64', '65', '53', '63', '80', '69', '71', '73',
                                    '61', '66', '72', '70',
                                    '78', '68', '75', '76', '74', '79', '77', '83', '81', '82', '86', '-9999', '94',
                                    '85', '103', '99', '88',
                                    '90', '93', '98', '96', '84', '91', '87', '108', '97', '95', '-9999', '92', '89',
                                    '102', '105', '111', '121',
                                    '104', '109', '107', '100', '117', '123', '114', '106', '118', '101', '110', '112',
                                    '119', '125', '116',
                                    '128', '126', '122', '-9999', '113', '124', '135', '127', '129', '115', '132',
                                    '131', '139', '143', '120',
                                    '130', '133', '142', '134', '152', '138', '141', '140', '137', '148', '145', '149',
                                    '150', '163', '165',
                                    '153', '136', '147', '-9999', '144', '158', '155', '151', '154', '-9999', '159',
                                    '157', '166', '146', '-9999',
                                    '167', '171', '164', '160', '168', '162', '-9999', '161', '169', '156', '170',
                                    '174', '172', '176', '175',
                                    '173', '177']

    breed_popularity_2013 = pd.DataFrame(animal_breeds)
    breed_popularity_2013['Breed11'] = pd.DataFrame(animal_breeds)
    breed_popularity_2013['Breed_Popularity_2013'] = pd.DataFrame(breed_popularity_scores_2013)
    breed_popularity_2013.drop(0, axis=1, inplace=True)
    breed_popularity_2013['Breed11'] = breed_popularity_2013['Breed11'].str.upper()

    # 2014 Breed Popularity Calculation
    breed_popularity_scores_2014 = ['1', '2', '3', '4', '5', '9', '6', '7', '10', '8', '12', '13', '11', '14', '15',
                                    '16', '18', '19', '17', '22',
                                    '20', '23', '21', '25', '26', '27', '28', '24', '32', '30', '29', '34', '33', '35',
                                    '48', '36', '37', '40',
                                    '42', '39', '38', '41', '45', '44', '47', '46', '49', '43', '50', '51', '51', '60',
                                    '52', '57', '54', '53',
                                    '55', '56', '59', '58', '62', '68', '66', '63', '61', '67', '76', '75', '71', '72',
                                    '69', '74', '73', '70',
                                    '77', '65', '78', '84', '80', '79', '82', '83', '85', '81', '87', '31', '92', '86',
                                    '103', '102', '88', '90',
                                    '94', '104', '95', '91', '89', '93', '110', '99', '98', '111', '101', '97', '105',
                                    '100', '114', '108', '106',
                                    '109', '115', '116', '118', '125', '119', '122', '127', '112', '120', '121', '123',
                                    '129', '124', '130',
                                    '128', '139', '64', '126', '133', '138', '132', '131', '134', '135', '143', '142',
                                    '144', '136', '141', '148',
                                    '146', '149', '145', '150', '152', '137', '140', '147', '155', '154', '161', '163',
                                    '157', '166', '151',
                                    '158', '107', '160', '169', '153', '159', '165', '117', '168', '170', '172', '156',
                                    '113', '162', '173',
                                    '176', '164', '167', '175', '96', '177', '178', '171', '174', '182', '179', '180',
                                    '184', '181', '183']

    breed_popularity_2014 = pd.DataFrame(animal_breeds)
    breed_popularity_2014['Breed11'] = pd.DataFrame(animal_breeds)
    breed_popularity_2014['Breed_Popularity_2014'] = pd.DataFrame(breed_popularity_scores_2014)
    breed_popularity_2014.drop(0, axis=1, inplace=True)
    breed_popularity_2014['Breed11'] = breed_popularity_2014['Breed11'].str.upper()

    # 2015 Breed Popularity Calculation
    breed_popularity_scores_2015 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                    '16', '17', '18', '19',
                                    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33',
                                    '34', '35', '36', '37',
                                    '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '50',
                                    '51', '52', '53', '54',
                                    '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68',
                                    '69', '70', '71', '72',
                                    '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
                                    '87', '88', '89', '90',
                                    '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
                                    '104', '105', '106', '107',
                                    '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119',
                                    '120', '121', '122',
                                    '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134',
                                    '135', '136', '137',
                                    '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                                    '150', '151', '152',
                                    '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164',
                                    '165', '166', '167',
                                    '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
                                    '180', '181', '182',
                                    '183', '184']

    breed_popularity_2015 = pd.DataFrame(animal_breeds)
    breed_popularity_2015['Breed11'] = pd.DataFrame(animal_breeds)
    breed_popularity_2015['Breed_Popularity_2015'] = pd.DataFrame(breed_popularity_scores_2015)
    breed_popularity_2015.drop(0, axis=1, inplace=True)
    breed_popularity_2015['Breed11'] = breed_popularity_2015['Breed11'].str.upper()

    # Joining Breeds Popularity Scores to the dataset depending on the breeds
    data_set = pd.merge(data_set, breed_popularity_2013, on='Breed11', how='left')
    data_set = pd.merge(data_set, breed_popularity_2014, on='Breed11', how='left')
    data_set = pd.merge(data_set, breed_popularity_2015, on='Breed11', how='left')

    data_set['Breed_Popularity_2013'] = data_set['Breed_Popularity_2013'].fillna(-9999).astype(float)
    data_set['Breed_Popularity_2014'] = data_set['Breed_Popularity_2014'].fillna(-9999).astype(float)
    data_set['Breed_Popularity_2015'] = data_set['Breed_Popularity_2015'].fillna(-9999).astype(float)

    return data_set


train_original = pd.read_csv('../input/train.csv', delimiter=',')
test_original = pd.read_csv('../input/test.csv', delimiter=',')
pd.options.mode.chained_assignment = None
test_ID = test_original['ID']
test_original.drop('ID', axis=1, inplace=True)

data_pre_process_initial(train_original, test_original)

# Removing outcome type columns from training data set
Target = train_original['Target']
train_original.drop('OutcomeSubtype', axis=1, inplace=True)
train_original.drop('Aggressive', axis=1, inplace=True)
train_original.drop('At Vet', axis=1, inplace=True)
train_original.drop('Barn', axis=1, inplace=True)
train_original.drop('Behavior', axis=1, inplace=True)
train_original.drop('Court/Investigation', axis=1, inplace=True)
train_original.drop('Enroute', axis=1, inplace=True)
train_original.drop('Foster', axis=1, inplace=True)
train_original.drop('In Foster', axis=1, inplace=True)
train_original.drop('In Kennel', axis=1, inplace=True)
train_original.drop('In Surgery', axis=1, inplace=True)
train_original.drop('Medical', axis=1, inplace=True)
train_original.drop('Offsite', axis=1, inplace=True)
train_original.drop('Partner', axis=1, inplace=True)
train_original.drop('Rabies Risk', axis=1, inplace=True)
train_original.drop('SCRP', axis=1, inplace=True)
train_original.drop('Suffering', axis=1, inplace=True)
train_original.drop('OutcomeType', axis=1, inplace=True)
train_original.drop('Target', axis=1, inplace=True)

train_2 = breed_data_pre_process(train_original)
train_2 = pd.get_dummies(train_2,
                         columns=['month', 'day',
                                  'Is_Mixed', 'AgeuponOutcome_cat', 'sprayed', 'Type_sex', 'SexuponOutcome_cat',
                                  'AnimalType_cat'])

test_2 = breed_data_pre_process(test_original)
test_2 = pd.get_dummies(test_2,
                        columns=['month', 'day',
                                 'Is_Mixed', 'AgeuponOutcome_cat', 'sprayed', 'Type_sex', 'SexuponOutcome_cat',
                                 'AnimalType_cat'])

train_select = pd.DataFrame(train_2.select_dtypes(include=['float64', 'int64', 'int32', 'float32']))
test_select = pd.DataFrame(test_2.select_dtypes(include=['float64', 'int64', 'int32', 'float32']))

train_select = train_select.fillna(-9999)
test_select = test_select.fillna(-9999)

dtrain = xgb.DMatrix(train_select, Target, missing=-9999)
dtest = xgb.DMatrix(test_select, missing=-9999)
train_select.to_csv("Train.csv", index=False)
test_select.to_csv("Test.csv", index=False)
