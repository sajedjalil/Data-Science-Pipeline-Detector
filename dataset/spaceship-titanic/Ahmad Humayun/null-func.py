import inspect
import os

def null_converter(train_df):
    ##describe the data
    train_df.describe()
    
    ##check null values in column
    print(train_df.isnull().sum())
    
    ##check total null values
    print("Total NUll Values:",train_df.isnull().sum().sum())
    
    
    """
    As when CryoSleep is true then RoomService,FoodCourt,ShoppongMall,Spa and VRDeck should be
    zero but there are NULL values so we will change it to Zero
    """
    train_df.loc[train_df.CryoSleep==True,['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]=0
    
    
    """
    As CryoSleep has NUll values so by comparing it with luxury amenties like RoomService,FoodCourt,
    ShoppingMall,Spa,VRDeck when they are zero it would indicate that 
    they in CryoSleep I-e CryoSleep would be true for them
    
    """    
    ##We will make Cryosleep true from NULL when RoomService,FoodCourt,ShoppingMall,Spa,VRDeck are zero
    train_df.loc[train_df.query('RoomService ==0 & FoodCourt==0 & ShoppingMall==0 & Spa==0& VRDeck==0& CryoSleep!=False &CryoSleep!=True')
                                .index,'CryoSleep']=True
    
    """
    And for remaining CryoSleep when luxury amenties like RoomService,FoodCourt,ShoppingMall,Spa,VRDeck
    when **either of them is greater than zero** it would indicate
    that they in **not in CryoSleep** I-e CryoSleep would be False for them
    """    
    ## making Cryosleep **False** from NULL when either of luxury amenties are greater than zero
    train_df.loc[train_df.query('(RoomService>0 | FoodCourt>0 | ShoppingMall>0 | Spa>0| VRDeck>0)& CryoSleep!=False &CryoSleep!=True')
                                .index,'CryoSleep']=False
    
    
    """
    now by checking CryoSleep there are 11 remaining null Values and w.r.t luxury amenities
    like RoomService,FoodCourt,ShoppingMall,Spa,VRDeck **either of them is NULL while others are zero** so 
    by logic we will go with **making null luxury amenity zero** as
    other four luxury amenity are zero and then **make Cryosleep True** as all luxury amenities will be zero
    """
    
    ##making NUll luxury amenities equal to zero while CryoSleep is NULL
    train_df.loc[train_df.query('CryoSleep!=False &CryoSleep!=True')
                                .index,['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]=0
    
    ##Make remaining Cryosleep equal to True as luxury amenities is zero
    train_df.loc[train_df.query('CryoSleep!=False &CryoSleep!=True')
                                .index,'CryoSleep']=True

    
    
    ##now we will fill the null values of numeric variables, Age,RoomService,Spa,FoodCourt,ShoppingMall,VRDeck by its median
    train_df.fillna(train_df.median(numeric_only=True),inplace=True)

    
    ##now we will add spending column which is total of luxury amenities
    train_df['spending']=train_df['RoomService']+train_df['FoodCourt']+train_df['ShoppingMall']+train_df['Spa']+train_df['VRDeck']
    
    
    """
    By checking various configuration i saw there are total 115 VIP null values for HomePlanet **Earth** and
    among known VIP values there is **zero** VIP true for **Earth**  and among 115, 49 have 0 spending
    so we can make VIP False for them and among remaining 66,
    64 are less than spending mean so we will make VIP False for all null where HomePlanet is Earth
    """
    train_df.loc[train_df.query('VIP!=True & VIP!=False & `HomePlanet` =="Earth"')
                            .index,'VIP']=False
    
    
    """
    there are 42 VIP null values for HomePlanet Europa and among them 5 have spending value less than spending mean(2500)
    and are closer to 2000 so we will make VIP true for all when HomePlanet is **Europa**
    """
    
    train_df.loc[train_df.query('VIP!=True & VIP!=False & `HomePlanet` =="Europa"')
                            .index,'VIP']=True
    
    
    """
    there are 43 VIP null values for HomePlanet Mars and among them 24 have spending value **zero** so to make things simple
    we will make VIP null values False for when Spending is zero for Mars and remaining true
    """
    ##making VIP False for spending==0
    train_df.loc[train_df.query('VIP!=True & VIP!=False & `HomePlanet` =="Mars"& spending==0')
                            .index,'VIP']=False
    ##make VIP true for Spending>0
    train_df.loc[train_df.query('VIP!=True & VIP!=False & `HomePlanet` =="Mars"& spending>0')
                            .index,'VIP']=True
    
    """
    Now there are 3 remaining VIP null as
    there Homeplanet is null so we will make them False as there spending is low than 2500
    """
    train_df.loc[train_df.query('VIP!=True & VIP!=False')
                            .index,'VIP']=False

    
    """
    now we have NULL values in categorical variables we have few options to dea with it,
    i will choose 2 methods I-e drop all NULL values and
    replace NULL values with new value. we will then compare which gives us good results
    """
    
    ### drop all null values, option 1
    train_drop_nan=train_df.dropna()
    
    ## check total null values for option 1
    print("null values w.r.t option 1",train_drop_nan.isnull().sum().sum())
    
    
    ### creating dummy variables for null values
    train_df.loc[train_df.query('`HomePlanet`!="Earth" &`HomePlanet`!="Europa" &`HomePlanet`!="Mars"')
                                .index,'HomePlanet']="Unknown"

    train_df.loc[train_df[train_df.Cabin.isna()==True]
                                .index,'Cabin']="Not_sure"

    train_df.loc[train_df[train_df.Destination.isna()==True]
                                .index,'Destination']="not_known"

    train_df.loc[train_df[train_df.Name.isna()==True]
                                .index,'Name']="no_name"

    
    ##check total null values
    print("Total NUll Values w.r.t option 2:",train_df.isnull().sum().sum())
    
    return train_drop_nan,train_df
# function to write the definition of our function to the file