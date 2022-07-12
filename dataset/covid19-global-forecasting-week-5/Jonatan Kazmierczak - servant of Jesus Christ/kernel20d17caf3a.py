#
# What should you expect in the near future? - https://bit.ly/2KiuRoZ 
#
# Future of the U.S. - https://biblehub.com/daniel/11-45.htm
#
# Future of the New York City - https://biblehub.com/niv/revelation/18.htm 
#



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import csv

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


pathPrefix = '../input/covid19-global-forecasting-week-5/'
trainFileName = pathPrefix + 'train.csv'
testFileName = pathPrefix + 'test.csv'
submissionFileName = 'submission.csv'

stats = dict( Fatalities = {}, ConfirmedCases = {} )
results = []

def read_train():
    with open(trainFileName) as file:
        reader = csv.DictReader(file)
        for r in reader:
            date = int( r['Date'].replace('-', '') )
            if date > 20200428:
                #print( date )
                category = r['Target']
                key = '{}_{}_{}'.format(
                    r['Country_Region'], r['Province_State'], r['County'] )
                value = float( r['TargetValue'] )
                if value < 0: value = 0.0
                if not key in stats[ category ]:
                    stats[ category ][ key ] = []
                values = stats[ category ][ key ]
                values.append( value )
                values.sort()
                
            #if r['Country_Region'] != 'Afghanistan': return 0


def process_test():
    with open(testFileName) as file:
        reader = csv.DictReader(file)
        for r in reader:
            #if r['Country_Region'] != 'Afghanistan': return 0
            
            fid = int( r['ForecastId'] )
            category = r['Target']
            key = '{}_{}_{}'.format(
                r['Country_Region'], r['Province_State'], r['County'] )
            values = stats[ category ][ key ]
            
            res = ( fid, values[0], values[ (len(values) >> 1) - 1 ], values[-1] )
            results.append(res)

def transform_result(fid, q, value):
    return '{}_{},{}\n'.format(fid, q, value)

def write_submission():
    with open(submissionFileName, 'w') as file:
        file.write('ForecastId_Quantile,TargetValue\n')
        
        for res in results:
            file.write( transform_result( res[0], 0.05, res[1] ) )
            file.write( transform_result( res[0], 0.5,  res[2] ) )
            file.write( transform_result( res[0], 0.95, res[3] ) )



read_train()
process_test()
write_submission()

print( stats['ConfirmedCases']['Switzerland__'] )
print( stats['Fatalities']['Switzerland__'] )
print()
print( results[20790] )
print( results[20791] )
