import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
 
# path1 = '/home/jian/DATA_SETS/kaggle/forests/train.csv'
# path2 = '/home/jian/DATA_SETS/kaggle/forests/test.csv'

path1 ='../input/train.csv'
path2 ='../input/test.csv'

 
def preprocess(data2):
  
  data = data2
  feature_cols_for_filling_missing= [col for col in data.columns if col  not in ['Hillshade_3pm', 'Id']]
  X_train = data[feature_cols_for_filling_missing][data.Hillshade_3pm!=0]
  y_train=data['Hillshade_3pm'][data.Hillshade_3pm!=0]
  X_test=data[feature_cols_for_filling_missing][data.Hillshade_3pm==0]
  from sklearn.ensemble import RandomForestRegressor 

  rfg = RandomForestRegressor()
  rfg.fit(X_train, y_train)
  data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=np.around(rfg.predict(X_test))
  return data 



def feature_engineering(data2):
    
    
    data = data2
    
    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology
         
    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology
     
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
     
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']
     
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']
     
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']
     
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']
    
    data['Soil']=0
    for i in range(1,41):
      data['Soil']=data['Soil']+i*data['Soil_Type'+str(i)]
      
     
    data['Wilderness_Area']=0
    for i in range(1,5):
      data['Wilderness_Area']=data['Wilderness_Area']+i*data['Wilderness_Area'+str(i)]
      
    return data
  
def get_features():
    return ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points',
    'Ele_minus_VDtHyd','Ele_plus_VDtHyd','Distanse_to_Hydrolody','Hydro_plus_Fire','Hydro_minus_Fire','Hydro_plus_Road',
    'Hydro_minus_Road','Fire_plus_Road','Fire_minus_Road','Soil','Wilderness_Area']
 
def main():    
  
  train_df = pd.read_csv(path1)
  test_df = pd.read_csv(path2)

  train_df = preprocess(train_df)
  test_df = preprocess(test_df)
   
  train_df = feature_engineering(train_df)
  test_df = feature_engineering(test_df)
  
  features = get_features()

  y_train = train_df['Cover_Type'].values
  test_id = test_df['Id']
  X_train =  train_df[:][features].values
  
  #X_test = test_df[:][features]

  
  def split_X_test():
    
    length = len(test_df)
    n  =  length // 10000
    print(n)
    split_test_data = []
    
    for i in range(n):
    
      split_test_data.append(test_df[i*10000:(i+1)*10000][features].values)
    
    split_test_data.append(test_df[n*10000:length][features].values)
      
    return split_test_data
 
  X_test = split_X_test()
 
  print('Start')
  
  clf = ExtraTreesClassifier(max_features=0.3, n_estimators=500)
  clf.fit(X_train, y_train)

  print('train over')
  
  y_predict = []
   
  for var in X_test:
 
    y_predict.extend(clf.predict(var))
  
  
 
  submission = pd.DataFrame(data= {'Id' : test_id, 'Cover_Type': y_predict})
  submission.to_csv('result.csv', index=False)
 
  print('Compelte')

if __name__ == '__main__':
    main()
