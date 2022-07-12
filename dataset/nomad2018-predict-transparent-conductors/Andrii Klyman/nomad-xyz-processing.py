import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
# ---------------------------------------------------
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def read_all_xyz_data(my_dataset,train=1):
    ga_cols = []; al_cols = []; in_cols = []; o_cols = [];   
    for my_ in range(6):  ga_cols.append('ga_'+str(my_)); 
    for my_ in range(6):  al_cols.append('al_'+str(my_)); 
    for my_ in range(6):  in_cols.append('in_'+str(my_)); 
    for my_ in range(6):  o_cols.append('o_'+str(my_)); 
   
    ga_df = pd.DataFrame(columns=ga_cols)
    al_df = pd.DataFrame(columns=al_cols)
    in_df = pd.DataFrame(columns=in_cols)
    o_df  = pd.DataFrame(columns= o_cols)
    # -----------------------------------------------    
    if train == 1:
        Local_ids  = my_dataset.id.values
        Local_Path = "../input/train/{}/geometry.xyz"
    if train == 0:
        Local_ids = my_dataset.id.values
        Local_Path = "../input/test/{}/geometry.xyz"
    # -----------------------------------------------
    print(Local_ids)
    for i in Local_ids: 
        #if i==6: break; # для тестов и проверок пока берем 5 файликов
        filename = Local_Path.format(i)
        print("--------------------> "+filename)
        ga_list = []
        al_list = []
        o_list  = []
        in_list = []
        # -----------------------------------------------    
        with open(filename) as f:
            for line in f.readlines():
                if line.rfind('atom')==-1: continue # пропускаем шапку
                x = line.split(' ')
                # print('-->',x[4],'--')
                # группируем атомы в группы
                if line.rfind("Ga")!=-1: ga_list.append(np.array(x[1:4], dtype=np.float))
                if line.rfind("Al")!=-1: al_list.append(np.array(x[1:4], dtype=np.float))
                if line.rfind("In")!=-1: in_list.append(np.array(x[1:4], dtype=np.float))
                if line.rfind("O") !=-1:  o_list.append(np.array(x[1:4], dtype=np.float))
        # -----------------------------------------------------------    
        # -- тут понижаем размерность для каждой группы точек
        # пример результата 
        #[[-1.52497367  1.67679902]
        # [ 7.94587215 -0.57142152]
        # [-6.42089847 -1.1053775 ]]    
        for mmy_ in range(4):
            if mmy_==0: my_list = ga_list; my_df = ga_df;
            if mmy_==1: my_list = al_list; my_df = al_df;
            if mmy_==2: my_list = in_list; my_df = in_df;
            if mmy_==3: my_list = o_list;  my_df = o_df;
            # ---------------------------------------------
            if (len(my_list)<2): temp_my=[0,0,0,0,0,0]
            else:
                model = PCA(n_components=2)
                my_list = np.array(my_list)
                temp_my = model.fit_transform(my_list.transpose())
                # проходим по 2м циклам: вложенному списку и внешнему
                temp_my = [item for sublist in temp_my for item in sublist]
            my_df.loc[i] = temp_my
            #print('=>',my_df.head())
    # -----------------------------------------------
    # подцепляем эти датафреймы к оригинальному
    ga_df["id"] = ga_df.index
    my_dataset = pd.merge(my_dataset, ga_df, on='id',how = "left")
    al_df["id"] = al_df.index
    my_dataset = pd.merge(my_dataset, al_df, on='id',how = "left")
    in_df["id"] = in_df.index
    my_dataset = pd.merge(my_dataset, in_df, on='id',how = "left")
    o_df["id"] = o_df.index
    my_dataset = pd.merge(my_dataset, o_df, on='id',how = "left")
    my_dataset[['in_0','in_1','in_2','in_3','in_4','in_5']] = my_dataset[['in_0','in_1','in_2','in_3','in_4','in_5']].astype(float)
    return my_dataset
# ---------------------------------------------------

new_train = read_all_xyz_data(train,1)
#print('=>',new_train.head())


new_test = read_all_xyz_data(test,0)
#print('=>',new_test.head())
print(new_test.dtypes)

#print(train.head(5))











