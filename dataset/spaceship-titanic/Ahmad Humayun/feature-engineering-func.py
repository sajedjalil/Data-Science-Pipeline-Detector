# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-03-11T10:53:15.248069Z","iopub.execute_input":"2022-03-11T10:53:15.248477Z","iopub.status.idle":"2022-03-11T10:53:15.280515Z","shell.execute_reply.started":"2022-03-11T10:53:15.248375Z","shell.execute_reply":"2022-03-11T10:53:15.279647Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:53:17.253604Z","iopub.execute_input":"2022-03-11T10:53:17.253890Z","iopub.status.idle":"2022-03-11T10:53:17.258912Z","shell.execute_reply.started":"2022-03-11T10:53:17.253847Z","shell.execute_reply":"2022-03-11T10:53:17.257936Z"},"jupyter":{"outputs_hidden":false}}
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Feature Engineering
def feature_engineering_function(data):

# %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:53:44.474955Z","iopub.execute_input":"2022-03-11T10:53:44.475171Z","iopub.status.idle":"2022-03-11T10:53:44.499409Z","shell.execute_reply.started":"2022-03-11T10:53:44.475144Z","shell.execute_reply":"2022-03-11T10:53:44.498396Z"},"jupyter":{"outputs_hidden":false}}
    data.head()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:26.919470Z","iopub.execute_input":"2022-03-11T10:54:26.919833Z","iopub.status.idle":"2022-03-11T10:54:26.963076Z","shell.execute_reply.started":"2022-03-11T10:54:26.919797Z","shell.execute_reply":"2022-03-11T10:54:26.962222Z"},"jupyter":{"outputs_hidden":false}}
    data['HomePlanet'].replace(['Europa','Earth','Mars'],[0,1,2],inplace=True)
    data['CryoSleep'].replace([False,True],[0,1],inplace=True)
    data['Destination'].replace(['TRAPPIST-1e','PSO J318.5-22','55 Cancri e'],[0,1,2],inplace=True)
    data['VIP'].replace([False,True],[0,1],inplace=True)

    print("Min Age:",data.Age.min())
    print("Mean Age:",data.Age.mean())
    print("Max Age:",data.Age.max())
    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:28.583088Z","iopub.execute_input":"2022-03-11T10:54:28.583425Z","iopub.status.idle":"2022-03-11T10:54:28.602983Z","shell.execute_reply.started":"2022-03-11T10:54:28.583387Z","shell.execute_reply":"2022-03-11T10:54:28.601502Z"},"jupyter":{"outputs_hidden":false}}
    
    data['Age_band']=0
    data.loc[data['Age']<=16,'Age_band']=0
    data.loc[(data['Age']>16) & (data['Age']<=32),'Age_band']=1
    data.loc[(data['Age']>32) & (data['Age']<=48),'Age_band']=2
    data.loc[(data['Age']>48) & (data['Age']<=64),'Age_band']=3
    data.loc[(data['Age']>64), 'Age_band']=4

    # %% [markdown]
    # when i used 10 partitions then the coorelation value i got between Age and Age_band was around 0.8 but when i used 5 partitions the coorelation value i got was 0.95 so went with 5 partitions

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:30.775469Z","iopub.execute_input":"2022-03-11T10:54:30.775973Z","iopub.status.idle":"2022-03-11T10:54:30.783212Z","shell.execute_reply.started":"2022-03-11T10:54:30.775936Z","shell.execute_reply":"2022-03-11T10:54:30.782129Z"},"jupyter":{"outputs_hidden":false}}
    print(data['Age_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:32.333451Z","iopub.execute_input":"2022-03-11T10:54:32.334401Z","iopub.status.idle":"2022-03-11T10:54:32.342357Z","shell.execute_reply.started":"2022-03-11T10:54:32.334357Z","shell.execute_reply":"2022-03-11T10:54:32.341045Z"},"jupyter":{"outputs_hidden":false}}
    print("Min RoomService:",data.RoomService.min())
    print("Mean RoomService:",data.RoomService.mean())
    print("Max RoomService:",data.RoomService.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:33.005172Z","iopub.execute_input":"2022-03-11T10:54:33.005472Z","iopub.status.idle":"2022-03-11T10:54:33.028732Z","shell.execute_reply.started":"2022-03-11T10:54:33.005441Z","shell.execute_reply":"2022-03-11T10:54:33.027303Z"},"jupyter":{"outputs_hidden":false}}
    data['RoomService_band']=0
    data.loc[data['RoomService']<=1000,'RoomService_band']=0
    data.loc[(data['RoomService']>1000) & (data['RoomService']<=2000),'RoomService_band']=1
    data.loc[(data['RoomService']>2000) & (data['RoomService']<=3000),'RoomService_band']=2
    data.loc[(data['RoomService']>3000) & (data['RoomService']<=4000),'RoomService_band']=3
    data.loc[(data['RoomService']>4000) & (data['RoomService']<=5000),'RoomService_band']=4
    data.loc[(data['RoomService']>5000) & (data['RoomService']<=6000),'RoomService_band']=5
    data.loc[(data['RoomService']>6000) & (data['RoomService']<=7000),'RoomService_band']=6
    data.loc[(data['RoomService']>7000) & (data['RoomService']<=8000),'RoomService_band']=7
    data.loc[(data['RoomService']>8000) & (data['RoomService']<=9000),'RoomService_band']=8
    data.loc[(data['RoomService']>7200), 'RoomService_band']=9

    # %% [markdown]
    # when i used 5 partitions then the coorelation value i got between RoomService and RoomService_band was around 0.86 but when i used 10 partitions the coorelation value i got was 0.94 so went with 10 partitions

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:34.183978Z","iopub.execute_input":"2022-03-11T10:54:34.184283Z","iopub.status.idle":"2022-03-11T10:54:34.193192Z","shell.execute_reply.started":"2022-03-11T10:54:34.184252Z","shell.execute_reply":"2022-03-11T10:54:34.192483Z"},"jupyter":{"outputs_hidden":false}}
    print(data['RoomService_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:34.812317Z","iopub.execute_input":"2022-03-11T10:54:34.812699Z","iopub.status.idle":"2022-03-11T10:54:34.820165Z","shell.execute_reply.started":"2022-03-11T10:54:34.812658Z","shell.execute_reply":"2022-03-11T10:54:34.819570Z"},"jupyter":{"outputs_hidden":false}}
    print("Min FoodCourt:",data.FoodCourt.min())
    print("Mean FoodCourt:",data.FoodCourt.mean())
    print("Max FoodCourt:",data.FoodCourt.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:35.055275Z","iopub.execute_input":"2022-03-11T10:54:35.055580Z","iopub.status.idle":"2022-03-11T10:54:35.077080Z","shell.execute_reply.started":"2022-03-11T10:54:35.055539Z","shell.execute_reply":"2022-03-11T10:54:35.076205Z"},"jupyter":{"outputs_hidden":false}}
    data['FoodCourt_band']=0
    data.loc[data['FoodCourt']<=1000,'FoodCourt_band']=0
    data.loc[(data['FoodCourt']>1000) & (data['FoodCourt']<=2000),'FoodCourt_band']=1
    data.loc[(data['FoodCourt']>2000) & (data['FoodCourt']<=3000),'FoodCourt_band']=2
    data.loc[(data['FoodCourt']>3000) & (data['FoodCourt']<=4000),'FoodCourt_band']=3
    data.loc[(data['FoodCourt']>4000) & (data['FoodCourt']<=5000),'FoodCourt_band']=4
    data.loc[(data['FoodCourt']>5000) & (data['FoodCourt']<=6000),'FoodCourt_band']=5
    data.loc[(data['FoodCourt']>6000) & (data['FoodCourt']<=7000),'FoodCourt_band']=6
    data.loc[(data['FoodCourt']>7000) & (data['FoodCourt']<=8000),'FoodCourt_band']=7
    data.loc[(data['FoodCourt']>8000) & (data['FoodCourt']<=9000),'FoodCourt_band']=8
    data.loc[(data['FoodCourt']>7200), 'FoodCourt_band']=9

    # %% [markdown]
    # For 10 partitions the coorelation value i got between FoodCourt and FoodCourt_band was around 0.94

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:35.270508Z","iopub.execute_input":"2022-03-11T10:54:35.271380Z","iopub.status.idle":"2022-03-11T10:54:35.280136Z","shell.execute_reply.started":"2022-03-11T10:54:35.271331Z","shell.execute_reply":"2022-03-11T10:54:35.279368Z"},"jupyter":{"outputs_hidden":false}}
    print(data['FoodCourt_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:35.541335Z","iopub.execute_input":"2022-03-11T10:54:35.541857Z","iopub.status.idle":"2022-03-11T10:54:35.550466Z","shell.execute_reply.started":"2022-03-11T10:54:35.541816Z","shell.execute_reply":"2022-03-11T10:54:35.549855Z"},"jupyter":{"outputs_hidden":false}}
    print("Min ShoppingMall:",data.ShoppingMall.min())
    print("Mean ShoppingMall:",data.ShoppingMall.mean())
    print("Max ShoppingMall:",data.ShoppingMall.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:36.333925Z","iopub.execute_input":"2022-03-11T10:54:36.334445Z","iopub.status.idle":"2022-03-11T10:54:36.358614Z","shell.execute_reply.started":"2022-03-11T10:54:36.334391Z","shell.execute_reply":"2022-03-11T10:54:36.357758Z"},"jupyter":{"outputs_hidden":false}}
    data['ShoppingMall_band']=0
    data.loc[data['ShoppingMall']<=1000,'ShoppingMall_band']=0
    data.loc[(data['ShoppingMall']>1000) & (data['ShoppingMall']<=2000),'ShoppingMall_band']=1
    data.loc[(data['ShoppingMall']>2000) & (data['ShoppingMall']<=3000),'ShoppingMall_band']=2
    data.loc[(data['ShoppingMall']>3000) & (data['ShoppingMall']<=4000),'ShoppingMall_band']=3
    data.loc[(data['ShoppingMall']>4000) & (data['ShoppingMall']<=5000),'ShoppingMall_band']=4
    data.loc[(data['ShoppingMall']>5000) & (data['ShoppingMall']<=6000),'ShoppingMall_band']=5
    data.loc[(data['ShoppingMall']>6000) & (data['ShoppingMall']<=7000),'ShoppingMall_band']=6
    data.loc[(data['ShoppingMall']>7000) & (data['ShoppingMall']<=8000),'ShoppingMall_band']=7
    data.loc[(data['ShoppingMall']>8000) & (data['ShoppingMall']<=9000),'ShoppingMall_band']=8
    data.loc[(data['ShoppingMall']>7200), 'ShoppingMall_band']=9

    # %% [markdown]
    # For 10 partitions the coorelation value i got between ShoppingMall and ShoppingMall_band was around 0.92

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:36.749354Z","iopub.execute_input":"2022-03-11T10:54:36.750526Z","iopub.status.idle":"2022-03-11T10:54:36.760002Z","shell.execute_reply.started":"2022-03-11T10:54:36.750458Z","shell.execute_reply":"2022-03-11T10:54:36.758905Z"},"jupyter":{"outputs_hidden":false}}
    print(data['ShoppingMall_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:37.167642Z","iopub.execute_input":"2022-03-11T10:54:37.167946Z","iopub.status.idle":"2022-03-11T10:54:37.176022Z","shell.execute_reply.started":"2022-03-11T10:54:37.167913Z","shell.execute_reply":"2022-03-11T10:54:37.174833Z"},"jupyter":{"outputs_hidden":false}}
    print("Min Spa:",data.Spa.min())
    print("Mean Spa:",data.Spa.mean())
    print("Max Spa:",data.Spa.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:37.470334Z","iopub.execute_input":"2022-03-11T10:54:37.471319Z","iopub.status.idle":"2022-03-11T10:54:37.496509Z","shell.execute_reply.started":"2022-03-11T10:54:37.471269Z","shell.execute_reply":"2022-03-11T10:54:37.495886Z"},"jupyter":{"outputs_hidden":false}}
    data['Spa_band']=0
    data.loc[data['Spa']<=1000,'Spa_band']=0
    data.loc[(data['Spa']>1000) & (data['Spa']<=2000),'Spa_band']=1
    data.loc[(data['Spa']>2000) & (data['Spa']<=3000),'Spa_band']=2
    data.loc[(data['Spa']>3000) & (data['Spa']<=4000),'Spa_band']=3
    data.loc[(data['Spa']>4000) & (data['Spa']<=5000),'Spa_band']=4
    data.loc[(data['Spa']>5000) & (data['Spa']<=6000),'Spa_band']=5
    data.loc[(data['Spa']>6000) & (data['Spa']<=7000),'Spa_band']=6
    data.loc[(data['Spa']>7000) & (data['Spa']<=8000),'Spa_band']=7
    data.loc[(data['Spa']>8000) & (data['Spa']<=9000),'Spa_band']=8
    data.loc[(data['Spa']>7200), 'Spa_band']=9

    # %% [markdown]
    # For 10 partitions the coorelation value i got between Spa and Spa_band was around 0.95

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:38.912038Z","iopub.execute_input":"2022-03-11T10:54:38.912490Z","iopub.status.idle":"2022-03-11T10:54:38.921578Z","shell.execute_reply.started":"2022-03-11T10:54:38.912457Z","shell.execute_reply":"2022-03-11T10:54:38.920693Z"},"jupyter":{"outputs_hidden":false}}
    print(data['Spa_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:39.400882Z","iopub.execute_input":"2022-03-11T10:54:39.401370Z","iopub.status.idle":"2022-03-11T10:54:39.408628Z","shell.execute_reply.started":"2022-03-11T10:54:39.401338Z","shell.execute_reply":"2022-03-11T10:54:39.407763Z"},"jupyter":{"outputs_hidden":false}}
    print("Min VRDeck:",data.VRDeck.min())
    print("Mean VRDeck:",data.VRDeck.mean())
    print("Max VRDeck:",data.VRDeck.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:40.149317Z","iopub.execute_input":"2022-03-11T10:54:40.149676Z","iopub.status.idle":"2022-03-11T10:54:40.172514Z","shell.execute_reply.started":"2022-03-11T10:54:40.149645Z","shell.execute_reply":"2022-03-11T10:54:40.171660Z"},"jupyter":{"outputs_hidden":false}}
    data['VRDeck_band']=0
    data.loc[data['VRDeck']<=1000,'VRDeck_band']=0
    data.loc[(data['VRDeck']>1000) & (data['VRDeck']<=2000),'VRDeck_band']=1
    data.loc[(data['VRDeck']>2000) & (data['VRDeck']<=3000),'VRDeck_band']=2
    data.loc[(data['VRDeck']>3000) & (data['VRDeck']<=4000),'VRDeck_band']=3
    data.loc[(data['VRDeck']>4000) & (data['VRDeck']<=5000),'VRDeck_band']=4
    data.loc[(data['VRDeck']>5000) & (data['VRDeck']<=6000),'VRDeck_band']=5
    data.loc[(data['VRDeck']>6000) & (data['VRDeck']<=7000),'VRDeck_band']=6
    data.loc[(data['VRDeck']>7000) & (data['VRDeck']<=8000),'VRDeck_band']=7
    data.loc[(data['VRDeck']>8000) & (data['VRDeck']<=9000),'VRDeck_band']=8
    data.loc[(data['VRDeck']>7200), 'VRDeck_band']=9

    # %% [markdown]
    # For 10 partitions the coorelation value i got between VRDeck and VRDeck_band was around 0.96

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:40.679201Z","iopub.execute_input":"2022-03-11T10:54:40.679684Z","iopub.status.idle":"2022-03-11T10:54:40.686714Z","shell.execute_reply.started":"2022-03-11T10:54:40.679650Z","shell.execute_reply":"2022-03-11T10:54:40.686016Z"},"jupyter":{"outputs_hidden":false}}
    print(data['VRDeck_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:41.031507Z","iopub.execute_input":"2022-03-11T10:54:41.032019Z","iopub.status.idle":"2022-03-11T10:54:41.040254Z","shell.execute_reply.started":"2022-03-11T10:54:41.031980Z","shell.execute_reply":"2022-03-11T10:54:41.038562Z"},"jupyter":{"outputs_hidden":false}}
    print("Min spending:",data.spending.min())
    print("Mean spending:",data.spending.mean())
    print("Max spending:",data.spending.max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:41.775755Z","iopub.execute_input":"2022-03-11T10:54:41.776681Z","iopub.status.idle":"2022-03-11T10:54:41.798922Z","shell.execute_reply.started":"2022-03-11T10:54:41.776640Z","shell.execute_reply":"2022-03-11T10:54:41.797735Z"},"jupyter":{"outputs_hidden":false}}
    data['spending_band']=0
    data.loc[data['spending']<=1000,'spending_band']=0
    data.loc[(data['spending']>1000) & (data['spending']<=2000),'spending_band']=1
    data.loc[(data['spending']>2000) & (data['spending']<=3000),'spending_band']=2
    data.loc[(data['spending']>3000) & (data['spending']<=4000),'spending_band']=3
    data.loc[(data['spending']>4000) & (data['spending']<=5000),'spending_band']=4
    data.loc[(data['spending']>5000) & (data['spending']<=6000),'spending_band']=5
    data.loc[(data['spending']>6000) & (data['spending']<=7000),'spending_band']=6
    data.loc[(data['spending']>7000) & (data['spending']<=8000),'spending_band']=7
    data.loc[(data['spending']>8000) & (data['spending']<=9000),'spending_band']=8
    data.loc[(data['spending']>7200), 'spending_band']=9

    # %% [markdown]
    # For 10 partitions the coorelation value i got between spending and spending_band was around 0.91

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:57.305286Z","iopub.execute_input":"2022-03-11T10:54:57.305606Z","iopub.status.idle":"2022-03-11T10:54:57.313183Z","shell.execute_reply.started":"2022-03-11T10:54:57.305559Z","shell.execute_reply":"2022-03-11T10:54:57.312395Z"},"jupyter":{"outputs_hidden":false}}
    print(data['spending_band'].value_counts())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T10:54:47.746938Z","iopub.execute_input":"2022-03-11T10:54:47.747987Z","iopub.status.idle":"2022-03-11T10:54:47.777868Z","shell.execute_reply.started":"2022-03-11T10:54:47.747891Z","shell.execute_reply":"2022-03-11T10:54:47.777019Z"},"jupyter":{"outputs_hidden":false}}
    data.head()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T11:00:11.113394Z","iopub.execute_input":"2022-03-11T11:00:11.114577Z","iopub.status.idle":"2022-03-11T11:00:11.131935Z","shell.execute_reply.started":"2022-03-11T11:00:11.114488Z","shell.execute_reply":"2022-03-11T11:00:11.130905Z"},"jupyter":{"outputs_hidden":false}}
    data.drop(['Name','Age','RoomService','FoodCourt','Spa','ShoppingMall','VRDeck','spending','Cabin'],axis=1,inplace=True)

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-11T11:00:12.471559Z","iopub.execute_input":"2022-03-11T11:00:12.472496Z","iopub.status.idle":"2022-03-11T11:00:13.512999Z","shell.execute_reply.started":"2022-03-11T11:00:12.472457Z","shell.execute_reply":"2022-03-11T11:00:13.511797Z"},"jupyter":{"outputs_hidden":false}}
    fig = plt.figure(figsize=(18,8))

    sns.heatmap(data.corr(),annot=True)
    plt.title("Coorelation")
    plt.show()
    
    return data

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
