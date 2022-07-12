import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau

#Read and Prepare Data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
all_df=train_df.append(test_df,sort=False)
all_df=all_df.reset_index(drop=True)


cont_cols=[
        "Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
        "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"
        ]
        
#Add shade variation
all_df["daily_shade_sd"]=all_df.loc[:,["Hillshade_9am","Hillshade_3pm","Hillshade_Noon"]].apply(np.std,axis=1).tolist()
#Add Pythagorean distance to hydrology
all_df["hydro_dist"]=(all_df["Horizontal_Distance_To_Hydrology"]**2+all_df["Vertical_Distance_To_Hydrology"]**2)**.5
#Mean Distance To Key Points
all_df["Mean_Amenities"]=(all_df["Horizontal_Distance_To_Fire_Points"] + all_df["Horizontal_Distance_To_Hydrology"] + all_df["Horizontal_Distance_To_Roadways"]) / 3 
#Mean Distance to Fire and Water
all_df["Mean_Fire_Hyd"]=(all_df["Horizontal_Distance_To_Fire_Points"] + all_df["Horizontal_Distance_To_Hydrology"]) / 2 

#Normalize continuous columns
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(all_df[cont_cols+["daily_shade_sd","hydro_dist","Mean_Amenities","Mean_Fire_Hyd"]])
all_df[cont_cols+["daily_shade_sd","hydro_dist","Mean_Amenities","Mean_Fire_Hyd"]]=scaled_df

##Soil Types
soil_cols=all_df.loc[:,["Soil_Type"+str(x) for x in range(1,41)]]
all_df["Soil_Type"]=soil_cols.idxmax(axis=1).str.replace("\D","").astype("int").tolist()
#Lets frick around with the descriptions of the soil
soil_lab={
    1: "Cathedral family - Rock outcrop complex, extremely stony.",
    2: "Vanet - Ratake families complex, very stony.",
    3: "Haploborolis - Rock outcrop complex, rubbly.",
    4: "Ratake family - Rock outcrop complex, rubbly.",
    5: "Vanet family - Rock outcrop complex complex, rubbly.",
    6: "Vanet - Wetmore families - Rock outcrop complex, stony.",
    7: "Gothic family.",
    8: "Supervisor - Limber families complex.",
    9: "Troutville family, very stony.",
    10: "Bullwark - Catamount families - Rock outcrop complex, rubbly.",
    11: "Bullwark - Catamount families - Rock land complex, rubbly.",
    12: "Legault family - Rock land complex, stony.",
    13: "Catamount family - Rock land - Bullwark family complex, rubbly.",
    14: "Pachic Argiborolis - Aquolis complex.",
    15: "unspecified in the USFS Soil and ELU Survey.",
    16: "Cryaquolis - Cryoborolis complex.",
    17: "Gateview family - Cryaquolis complex.",
    18: "Rogert family, very stony.",
    19: "Typic Cryaquolis - Borohemists complex.",
    20: "Typic Cryaquepts - Typic Cryaquolls complex.",
    21: "Typic Cryaquolls - Leighcan family, till substratum complex.",
    22: "Leighcan family, till substratum, extremely bouldery.",
    23: "Leighcan family, till substratum - Typic Cryaquolls complex.",
    24: "Leighcan family, extremely stony.",
    25: "Leighcan family, warm, extremely stony.",
    26: "Granile - Catamount families complex, very stony.",
    27: "Leighcan family, warm - Rock outcrop complex, extremely stony.",
    28: "Leighcan family - Rock outcrop complex, extremely stony.",
    29: "Como - Legault families complex, extremely stony.",
    30: "Como family - Rock land - Legault family complex, extremely stony.",
    31: "Leighcan - Catamount families complex, extremely stony.",
    32: "Catamount family - Rock outcrop - Leighcan family complex, extremely stony.",
    33: "Leighcan - Catamount families - Rock outcrop complex, extremely stony.",
    34: "Cryorthents - Rock land complex, extremely stony.",
    35: "Cryumbrepts - Rock outcrop - Cryaquepts complex.",
    36: "Bross family - Rock land - Cryumbrepts complex, extremely stony.",
    37: "Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.",
    38: "Leighcan - Moran families - Cryaquolls complex, extremely stony.",
    39: "Moran family - Cryorthents - Leighcan family complex, extremely stony.",
    40: "Moran family - Cryorthents - Rock land complex, extremely stony."
        }

label_list=[x.replace("-","999").replace(",","999").split("999") for x in soil_lab.values()]
label_list=[[x.strip(".").strip(" ").lower() for x in line] for line in label_list]
label_list=[[x.replace(" ","_") for x in line] for line in label_list]
words=pd.Series([x for line in label_list for x in line]).unique()

word_frame=pd.DataFrame()
for word in words:
    word_frame["SL_"+word]=[1 if word in x else 0 for x in label_list]
word_frame["soil_lable"]=word_frame.index+1    
all_df=pd.merge(all_df,word_frame,left_on="Soil_Type",right_on="soil_lable")
del all_df["Soil_Type"]


#Split back to train and test
train_df=all_df.loc[~pd.isna(all_df.Cover_Type),:]
test_df=all_df.loc[pd.isna(all_df.Cover_Type),:]
x_train=train_df.loc[:,[x for x in train_df.columns if x not in ["Id","Cover_Type"]]]
x_test=test_df.loc[:,[x for x in test_df.columns if x not in ["Id","Cover_Type"]]]
#Train Val Split
y=pd.get_dummies(train_df["Cover_Type"])
y_order=list(y.columns)
X_train, X_val, y_train, y_val = train_test_split(
        x_train,y,
        test_size=0.05, random_state=42
        )

model=Sequential()
model.add(Dense(1000, activation = "relu",input_shape=(119,)))
model.add(Dense(750, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(7, activation = "softmax"))

##Compile
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['accuracy'])

#helps converge faster
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 75
batch_size=64

#fit model
history=model.fit(
        X_train,y_train,batch_size=batch_size,epochs = epochs,
        validation_data = (X_val,y_val),callbacks=[learning_rate_reduction],
        verbose=2
        )

classes=model.predict(x_test)
preds=np.argmax(classes,axis = 1)
preds=[int(y_order[x]) for x in preds]
pd.DataFrame({"Id":test_df["Id"].tolist(),"Cover_Type":preds}).to_csv("predictions.csv",index=False)
