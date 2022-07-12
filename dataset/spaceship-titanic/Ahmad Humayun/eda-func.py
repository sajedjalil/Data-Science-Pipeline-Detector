# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:29.019628Z","iopub.execute_input":"2022-03-10T08:03:29.020347Z","iopub.status.idle":"2022-03-10T08:03:29.039374Z","shell.execute_reply.started":"2022-03-10T08:03:29.020253Z","shell.execute_reply":"2022-03-10T08:03:29.038183Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:29.723699Z","iopub.execute_input":"2022-03-10T08:03:29.724286Z","iopub.status.idle":"2022-03-10T08:03:29.729528Z","shell.execute_reply.started":"2022-03-10T08:03:29.724249Z","shell.execute_reply":"2022-03-10T08:03:29.728103Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:30.150191Z","iopub.execute_input":"2022-03-10T08:03:30.150597Z","iopub.status.idle":"2022-03-10T08:03:30.219102Z","shell.execute_reply.started":"2022-03-10T08:03:30.150565Z","shell.execute_reply":"2022-03-10T08:03:30.218376Z"}}


# %% [markdown]
# imported utility script i made [here](https://www.kaggle.com/ahmadhumayun/null-func) for cleaning null values and notebook from which i made the utility script is [here](https://www.kaggle.com/ahmadhumayun/spaceship-titanic-null-values)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:30.583444Z","iopub.execute_input":"2022-03-10T08:03:30.583847Z","iopub.status.idle":"2022-03-10T08:03:30.590704Z","shell.execute_reply.started":"2022-03-10T08:03:30.583796Z","shell.execute_reply":"2022-03-10T08:03:30.589990Z"}}

# %% [markdown]
# cleaning null values function return 2 dataframes first dataframe is with NULL values dropped from it and second is with NULL values replaced with dummy values
# 
# NULL values dropped or changed to dummy values were only used on variable Cabin,HomePlanet,Destination,Name. Others were filled in both
# 
# for details you can check the notebook [here](https://www.kaggle.com/ahmadhumayun/spaceship-titanic-null-values)

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:30.841674Z","iopub.execute_input":"2022-03-10T08:03:30.842349Z","iopub.status.idle":"2022-03-10T08:03:30.974204Z","shell.execute_reply.started":"2022-03-10T08:03:30.842306Z","shell.execute_reply":"2022-03-10T08:03:30.973422Z"}}

# %% [markdown]
# ## EDA

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:31.148560Z","iopub.execute_input":"2022-03-10T08:03:31.149820Z","iopub.status.idle":"2022-03-10T08:03:31.675625Z","shell.execute_reply.started":"2022-03-10T08:03:31.149756Z","shell.execute_reply":"2022-03-10T08:03:31.674925Z"}}
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# i will do exploratory anlysis with dropped null values as it will be easier to visualize and get understanding of data

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:31.677010Z","iopub.execute_input":"2022-03-10T08:03:31.677817Z","iopub.status.idle":"2022-03-10T08:03:31.682068Z","shell.execute_reply.started":"2022-03-10T08:03:31.677779Z","shell.execute_reply":"2022-03-10T08:03:31.681237Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:31.683264Z","iopub.execute_input":"2022-03-10T08:03:31.683601Z","iopub.status.idle":"2022-03-10T08:03:31.713450Z","shell.execute_reply.started":"2022-03-10T08:03:31.683569Z","shell.execute_reply":"2022-03-10T08:03:31.712832Z"}}


# %% [markdown]
# ## HomePlanet

# %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:31.968923Z","iopub.execute_input":"2022-03-10T08:03:31.969302Z","iopub.status.idle":"2022-03-10T08:03:32.026710Z","shell.execute_reply.started":"2022-03-10T08:03:31.969255Z","shell.execute_reply":"2022-03-10T08:03:32.025727Z"}}
def EDA_function(data):    
    print(pd.crosstab(data['HomePlanet'],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:32.088252Z","iopub.execute_input":"2022-03-10T08:03:32.088516Z","iopub.status.idle":"2022-03-10T08:03:33.325316Z","shell.execute_reply.started":"2022-03-10T08:03:32.088488Z","shell.execute_reply":"2022-03-10T08:03:33.323862Z"}}
    fig = plt.figure(figsize=(20,14))
    plt.subplot(2, 2, 1)
    sns.barplot(x="HomePlanet",y='Transported',data=data)
    plt.title("Transported(True) Ratio w.r.t HomePlanet")
    plt.subplot(2, 2, 2)
    sns.countplot(x='HomePlanet',hue='Transported',data=data)
    plt.title("Transported Count w.r.t HomePlanet")
    plt.subplot(2, 2, 3)
    plt.pie(data.query('Transported==True').groupby('Transported')['HomePlanet'].value_counts(),autopct="%.1f%%",explode=[0.05]*3,labels=['Earth','Europa','Mars'],shadow=True)
    plt.title("HomePlanet percentage w.r.t Transported(True)")
    plt.subplot(2, 2, 4)
    plt.pie(data.query('Transported==False').groupby('Transported')['HomePlanet'].value_counts(),autopct="%.1f%%",explode=[0.05]*3,labels=['Earth','Mars','Europa'],shadow=True)
    plt.title("HomePlanet percentage w.r.t Not Transported(Falsse)")

    sns.catplot(x='HomePlanet',y='Transported',kind='point',data=data)
    plt.title("Transported(True) Ratio w.r.t HomePlanet")
    plt.show()
    plt.close()
    
    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:33.327082Z","iopub.execute_input":"2022-03-10T08:03:33.327790Z","iopub.status.idle":"2022-03-10T08:03:33.371924Z","shell.execute_reply.started":"2022-03-10T08:03:33.327753Z","shell.execute_reply":"2022-03-10T08:03:33.370926Z"}}
    print(pd.crosstab([data['HomePlanet'],data['CryoSleep']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:33.373242Z","iopub.execute_input":"2022-03-10T08:03:33.373454Z","iopub.status.idle":"2022-03-10T08:03:34.257131Z","shell.execute_reply.started":"2022-03-10T08:03:33.373428Z","shell.execute_reply":"2022-03-10T08:03:34.256174Z"}}

    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="HomePlanet",y='Transported',hue='CryoSleep',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t HomePlanet and CryoSleep")

    sns.pointplot(x='HomePlanet',y='Transported',hue='CryoSleep',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t HomePlanet and CryoSleep")

    plt.show()
    plt.close()



    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:34.259016Z","iopub.execute_input":"2022-03-10T08:03:34.259280Z","iopub.status.idle":"2022-03-10T08:03:34.307278Z","shell.execute_reply.started":"2022-03-10T08:03:34.259238Z","shell.execute_reply":"2022-03-10T08:03:34.306229Z"}}
    print(pd.crosstab([data['HomePlanet'],data['Destination']],data['Transported'],margins=True))


    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:34.308633Z","iopub.execute_input":"2022-03-10T08:03:34.308881Z","iopub.status.idle":"2022-03-10T08:03:35.380612Z","shell.execute_reply.started":"2022-03-10T08:03:34.308852Z","shell.execute_reply":"2022-03-10T08:03:35.379787Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="HomePlanet",y='Transported',hue='Destination',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t HomePlanet and Destination")

    sns.pointplot(x='HomePlanet',y='Transported',hue='Destination',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t HomePlanet and Destination")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:35.381799Z","iopub.execute_input":"2022-03-10T08:03:35.382021Z","iopub.status.idle":"2022-03-10T08:03:35.426357Z","shell.execute_reply.started":"2022-03-10T08:03:35.381992Z","shell.execute_reply":"2022-03-10T08:03:35.425771Z"}}
    print(pd.crosstab([data['HomePlanet'],data['VIP']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:35.427564Z","iopub.execute_input":"2022-03-10T08:03:35.427954Z","iopub.status.idle":"2022-03-10T08:03:36.360642Z","shell.execute_reply.started":"2022-03-10T08:03:35.427917Z","shell.execute_reply":"2022-03-10T08:03:36.359992Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="HomePlanet",y='Transported',hue='VIP',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t HomePlanet and VIP")

    sns.pointplot(x='HomePlanet',y='Transported',hue='VIP',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t HomePlanet and VIP")

    plt.show()
    plt.close()

    # %% [markdown]
    # ## CryoSleep

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:39.680132Z","iopub.execute_input":"2022-03-10T08:03:39.680849Z","iopub.status.idle":"2022-03-10T08:03:39.729199Z","shell.execute_reply.started":"2022-03-10T08:03:39.680797Z","shell.execute_reply":"2022-03-10T08:03:39.728304Z"}}
    print(pd.crosstab(data['CryoSleep'],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:39.974301Z","iopub.execute_input":"2022-03-10T08:03:39.977284Z","iopub.status.idle":"2022-03-10T08:03:41.117976Z","shell.execute_reply.started":"2022-03-10T08:03:39.977231Z","shell.execute_reply":"2022-03-10T08:03:41.116818Z"}}
    fig = plt.figure(figsize=(20,14))
    plt.subplot(2, 2, 1)
    sns.barplot(x="CryoSleep",y='Transported',data=data)
    plt.title("Transported(True) Ratio w.r.t CryoSleep")
    plt.subplot(2, 2, 2)
    sns.countplot(x='CryoSleep',hue='Transported',data=data)
    plt.title("Transported Count w.r.t CryoSleep")
    plt.subplot(2, 2, 3)
    plt.pie(data.query('Transported==True').groupby('Transported')['CryoSleep'].value_counts(),autopct="%.1f%%",explode=[0.05]*2,labels=['True','False'],shadow=True)
    plt.title("CryoSleep percentage w.r.t Transported(True)")
    plt.subplot(2, 2, 4)
    plt.pie(data.query('Transported==False').groupby('Transported')['CryoSleep'].value_counts(),autopct="%.1f%%",explode=[0.05]*2,labels=data['CryoSleep'].unique(),shadow=True)
    plt.title("CryoSleep percentage w.r.t Not Transported(False)")

    sns.catplot(x='CryoSleep',y='Transported',kind='point',data=data)
    plt.title("Transported(True) Ratio w.r.t CryoSleep")
    plt.show()
    plt.close()
    
    # %% [code]


    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:03:41.143967Z","iopub.execute_input":"2022-03-10T08:03:41.144620Z","iopub.status.idle":"2022-03-10T08:03:41.195150Z","shell.execute_reply.started":"2022-03-10T08:03:41.144575Z","shell.execute_reply":"2022-03-10T08:03:41.194061Z"}}
    print(pd.crosstab([data['CryoSleep'],data['HomePlanet']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:07.694381Z","iopub.execute_input":"2022-03-10T08:13:07.694882Z","iopub.status.idle":"2022-03-10T08:13:08.675078Z","shell.execute_reply.started":"2022-03-10T08:13:07.694836Z","shell.execute_reply":"2022-03-10T08:13:08.674209Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="CryoSleep",y='Transported',hue='HomePlanet',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t CryoSleep and HomePlanet")

    sns.pointplot(x='CryoSleep',y='Transported',hue='HomePlanet',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t CryoSleep and HomePlanet")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:08.676820Z","iopub.execute_input":"2022-03-10T08:13:08.677670Z","iopub.status.idle":"2022-03-10T08:13:08.729288Z","shell.execute_reply.started":"2022-03-10T08:13:08.677631Z","shell.execute_reply":"2022-03-10T08:13:08.728146Z"}}
    print(pd.crosstab([data['CryoSleep'],data['Destination']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:08.730506Z","iopub.execute_input":"2022-03-10T08:13:08.730868Z","iopub.status.idle":"2022-03-10T08:13:09.754594Z","shell.execute_reply.started":"2022-03-10T08:13:08.730831Z","shell.execute_reply":"2022-03-10T08:13:09.753233Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="CryoSleep",y='Transported',hue='Destination',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t CryoSleep and Destination")

    sns.pointplot(x='CryoSleep',y='Transported',hue='Destination',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t CryoSleep and Destination")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:09.757154Z","iopub.execute_input":"2022-03-10T08:13:09.757513Z","iopub.status.idle":"2022-03-10T08:13:09.809064Z","shell.execute_reply.started":"2022-03-10T08:13:09.757467Z","shell.execute_reply":"2022-03-10T08:13:09.808452Z"}}
    print(pd.crosstab([data['CryoSleep'],data['VIP']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:09.810363Z","iopub.execute_input":"2022-03-10T08:13:09.811264Z","iopub.status.idle":"2022-03-10T08:13:10.783690Z","shell.execute_reply.started":"2022-03-10T08:13:09.811217Z","shell.execute_reply":"2022-03-10T08:13:10.782802Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="CryoSleep",y='Transported',hue='VIP',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t CryoSleep and VIP")

    sns.pointplot(x='CryoSleep',y='Transported',hue='VIP',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t CryoSleep and VIP")

    plt.show()
    plt.close()

    # %% [markdown]
    # ## Destination

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:10.785196Z","iopub.execute_input":"2022-03-10T08:13:10.785948Z","iopub.status.idle":"2022-03-10T08:13:10.829052Z","shell.execute_reply.started":"2022-03-10T08:13:10.785906Z","shell.execute_reply":"2022-03-10T08:13:10.827818Z"}}
    print(pd.crosstab(data['Destination'],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:10.830684Z","iopub.execute_input":"2022-03-10T08:13:10.831534Z","iopub.status.idle":"2022-03-10T08:13:12.025361Z","shell.execute_reply.started":"2022-03-10T08:13:10.831498Z","shell.execute_reply":"2022-03-10T08:13:12.023999Z"}}
    fig = plt.figure(figsize=(20,14))
    plt.subplot(2, 2, 1)
    sns.barplot(x="Destination",y='Transported',data=data)
    plt.title("Transported(True) Ratio w.r.t Destination")
    plt.subplot(2, 2, 2)
    sns.countplot(x='Destination',hue='Transported',data=data)
    plt.title("Transported Count w.r.t Destination")
    plt.subplot(2, 2, 3)
    plt.pie(data.query('Transported==True').groupby('Transported')['Destination'].value_counts(),autopct="%.1f%%",explode=[0.05]*3,labels=['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'],shadow=True)
    plt.title("HomePlanet percentage w.r.t Transported(True)")
    plt.subplot(2, 2, 4)
    plt.pie(data.query('Transported==False').groupby('Transported')['Destination'].value_counts(),autopct="%.1f%%",explode=[0.05]*3,labels=['TRAPPIST-1e','55 Cancri e','PSO J318.5-22'],shadow=True)
    plt.title("HomePlanet percentage w.r.t Not Transported(Falsse)")

    sns.catplot(x='Destination',y='Transported',kind='point',data=data)
    plt.title("Transported(True) Ratio w.r.t Destination")
    plt.show()
    plt.close()
    
    # %% [code]


    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:16.477901Z","iopub.execute_input":"2022-03-10T08:13:16.478237Z","iopub.status.idle":"2022-03-10T08:13:16.537583Z","shell.execute_reply.started":"2022-03-10T08:13:16.478204Z","shell.execute_reply":"2022-03-10T08:13:16.536375Z"}}
    print(pd.crosstab([data['Destination'],data['HomePlanet']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:16.914567Z","iopub.execute_input":"2022-03-10T08:13:16.914854Z","iopub.status.idle":"2022-03-10T08:13:18.032806Z","shell.execute_reply.started":"2022-03-10T08:13:16.914821Z","shell.execute_reply":"2022-03-10T08:13:18.031961Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="Destination",y='Transported',hue='HomePlanet',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t Destination and HomePlanet")

    sns.pointplot(x='Destination',y='Transported',hue='HomePlanet',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t Destination and HomePlanet")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:18.035931Z","iopub.execute_input":"2022-03-10T08:13:18.036665Z","iopub.status.idle":"2022-03-10T08:13:18.084201Z","shell.execute_reply.started":"2022-03-10T08:13:18.036593Z","shell.execute_reply":"2022-03-10T08:13:18.083077Z"}}
    print(pd.crosstab([data['Destination'],data['CryoSleep']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:18.085591Z","iopub.execute_input":"2022-03-10T08:13:18.086007Z","iopub.status.idle":"2022-03-10T08:13:19.014403Z","shell.execute_reply.started":"2022-03-10T08:13:18.085973Z","shell.execute_reply":"2022-03-10T08:13:19.013258Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="Destination",y='Transported',hue='CryoSleep',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t Destination and CryoSleep")

    sns.pointplot(x='Destination',y='Transported',hue='CryoSleep',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t Destination and CryoSleep")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:19.016165Z","iopub.execute_input":"2022-03-10T08:13:19.016411Z","iopub.status.idle":"2022-03-10T08:13:19.063517Z","shell.execute_reply.started":"2022-03-10T08:13:19.016383Z","shell.execute_reply":"2022-03-10T08:13:19.062542Z"}}
    print(pd.crosstab([data['Destination'],data['VIP']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:19.064840Z","iopub.execute_input":"2022-03-10T08:13:19.065051Z","iopub.status.idle":"2022-03-10T08:13:19.937637Z","shell.execute_reply.started":"2022-03-10T08:13:19.065026Z","shell.execute_reply":"2022-03-10T08:13:19.936565Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="Destination",y='Transported',hue='VIP',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t Destination and VIP")

    sns.pointplot(x='Destination',y='Transported',hue='VIP',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t Destination and VIP")

    plt.show()
    plt.close()

    # %% [markdown]
    # ## VIP

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:19.940021Z","iopub.execute_input":"2022-03-10T08:13:19.940346Z","iopub.status.idle":"2022-03-10T08:13:19.983705Z","shell.execute_reply.started":"2022-03-10T08:13:19.940304Z","shell.execute_reply":"2022-03-10T08:13:19.982768Z"}}
    print(pd.crosstab(data['VIP'],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:20.018207Z","iopub.execute_input":"2022-03-10T08:13:20.018466Z","iopub.status.idle":"2022-03-10T08:13:20.960715Z","shell.execute_reply.started":"2022-03-10T08:13:20.018439Z","shell.execute_reply":"2022-03-10T08:13:20.959805Z"}}
    fig = plt.figure(figsize=(18,12))
    plt.subplot(2, 3, 1)
    sns.barplot(x="VIP",y='Transported',data=data)
    plt.title("Transported(True) Ratio w.r.t VIP")
    plt.subplot(2, 3, 2)
    sns.countplot(x='VIP',hue='Transported',data=data)
    plt.title("Transported Count w.r.t VIP")
    plt.subplot(2, 3, 3)
    plt.pie(data.query('Transported==True').groupby('Transported')['VIP'].value_counts(),autopct="%.1f%%",explode=[0.05]*2,labels=['False','True'],shadow=True)
    plt.title("VIP percentage w.r.t Transported(True)")
    plt.subplot(2, 3, 4)
    plt.pie(data.query('Transported==False').groupby('Transported')['VIP'].value_counts(),autopct="%.1f%%",explode=[0.05]*2,labels=['False','True'],shadow=True)
    plt.title("VIP percentage w.r.t Not Transported(False)")
    plt.subplot(2, 3, 5)
    sns.pointplot(x='VIP',y='Transported',data=data)
    plt.title("Transported(True) Ratio w.r.t VIP")
    plt.show()
    plt.close()
    
    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:20.962687Z","iopub.execute_input":"2022-03-10T08:13:20.963044Z","iopub.status.idle":"2022-03-10T08:13:21.012209Z","shell.execute_reply.started":"2022-03-10T08:13:20.962999Z","shell.execute_reply":"2022-03-10T08:13:21.011002Z"}}
    print(pd.crosstab([data['VIP'],data['HomePlanet']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:13:23.839244Z","iopub.execute_input":"2022-03-10T08:13:23.839543Z","iopub.status.idle":"2022-03-10T08:13:24.787912Z","shell.execute_reply.started":"2022-03-10T08:13:23.839510Z","shell.execute_reply":"2022-03-10T08:13:24.786487Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="VIP",y='Transported',hue='HomePlanet',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t VIP and HomePlanet")

    sns.pointplot(x='VIP',y='Transported',hue='HomePlanet',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t VIP and HomePlanet")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T07:50:43.638317Z","iopub.execute_input":"2022-03-10T07:50:43.638545Z","iopub.status.idle":"2022-03-10T07:50:44.578445Z","shell.execute_reply.started":"2022-03-10T07:50:43.638517Z","shell.execute_reply":"2022-03-10T07:50:44.577656Z"}}
    print(pd.crosstab([data['VIP'],data['CryoSleep']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:14:09.366522Z","iopub.execute_input":"2022-03-10T08:14:09.366829Z","iopub.status.idle":"2022-03-10T08:14:10.126450Z","shell.execute_reply.started":"2022-03-10T08:14:09.366800Z","shell.execute_reply":"2022-03-10T08:14:10.125497Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="VIP",y='Transported',hue='CryoSleep',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t VIP and CryoSleep")

    sns.pointplot(x='VIP',y='Transported',hue='CryoSleep',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t VIP and CryoSleep")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T07:50:44.610224Z","iopub.execute_input":"2022-03-10T07:50:44.61059Z","iopub.status.idle":"2022-03-10T07:50:44.66124Z","shell.execute_reply.started":"2022-03-10T07:50:44.610552Z","shell.execute_reply":"2022-03-10T07:50:44.660354Z"}}
    print(pd.crosstab([data['VIP'],data['Destination']],data['Transported'],margins=True))

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:14:45.298941Z","iopub.execute_input":"2022-03-10T08:14:45.299248Z","iopub.status.idle":"2022-03-10T08:14:46.434050Z","shell.execute_reply.started":"2022-03-10T08:14:45.299213Z","shell.execute_reply":"2022-03-10T08:14:46.432020Z"}}
    _,ax=plt.subplots(1,2,figsize=(16,7))
    sns.barplot(x="VIP",y='Transported',hue='Destination',data=data,ax=ax[0])
    ax[0].set_title("Transported w.r.t VIP and Destination")

    sns.pointplot(x='VIP',y='Transported',hue='Destination',data=data,ax=ax[1])
    ax[1].set_title("Transported w.r.t VIP and Destination")

    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:15:06.071711Z","iopub.execute_input":"2022-03-10T08:15:06.072290Z","iopub.status.idle":"2022-03-10T08:15:06.095926Z","shell.execute_reply.started":"2022-03-10T08:15:06.072253Z","shell.execute_reply":"2022-03-10T08:15:06.095070Z"}}
    # %% [markdown]
    # ## Age

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:28:24.711564Z","iopub.execute_input":"2022-03-10T08:28:24.711879Z","iopub.status.idle":"2022-03-10T08:28:24.721165Z","shell.execute_reply.started":"2022-03-10T08:28:24.711842Z","shell.execute_reply":"2022-03-10T08:28:24.720431Z"}}
    print("Min age ",data['Age'].min())
    print("Mean age ",data['Age'].mean())
    print("Max age ",data['Age'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:26:52.731496Z","iopub.execute_input":"2022-03-10T08:26:52.731846Z","iopub.status.idle":"2022-03-10T08:26:53.787524Z","shell.execute_reply.started":"2022-03-10T08:26:52.731816Z","shell.execute_reply":"2022-03-10T08:26:53.786309Z"}}
    fig = plt.figure(figsize=(18,8))
    sns.countplot(x="Age",data=data)
    plt.title("Age")
    plt.show()
    plt.close()

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:25:07.503223Z","iopub.execute_input":"2022-03-10T08:25:07.503497Z","iopub.status.idle":"2022-03-10T08:25:08.844477Z","shell.execute_reply.started":"2022-03-10T08:25:07.503469Z","shell.execute_reply":"2022-03-10T08:25:08.843837Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="Age", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t Age")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="Age", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t Age")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="Age", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t Age")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="Age", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t Age")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['Age'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t Age")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['Age'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t Age")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## RoomService

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:28:08.757459Z","iopub.execute_input":"2022-03-10T08:28:08.757792Z","iopub.status.idle":"2022-03-10T08:28:08.764158Z","shell.execute_reply.started":"2022-03-10T08:28:08.757758Z","shell.execute_reply":"2022-03-10T08:28:08.763496Z"}}
    print("Min RoomService ",data['RoomService'].min())
    print("Mean RoomService ",data['RoomService'].mean())
    print("Max RoomService ",data['RoomService'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:31:05.598261Z","iopub.execute_input":"2022-03-10T08:31:05.598556Z","iopub.status.idle":"2022-03-10T08:31:06.802807Z","shell.execute_reply.started":"2022-03-10T08:31:05.598525Z","shell.execute_reply":"2022-03-10T08:31:06.801600Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="RoomService", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t RoomService")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="RoomService", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t RoomService")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="RoomService", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t RoomService")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="RoomService", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t RoomService")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['RoomService'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t RoomService")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['RoomService'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t RoomService")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## FoodCourt

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:34:28.619357Z","iopub.execute_input":"2022-03-10T08:34:28.619758Z","iopub.status.idle":"2022-03-10T08:34:28.632667Z","shell.execute_reply.started":"2022-03-10T08:34:28.619708Z","shell.execute_reply":"2022-03-10T08:34:28.631960Z"}}
    print("Min FoodCourt ",data['FoodCourt'].min())
    print("Mean FoodCourt ",data['FoodCourt'].mean())
    print("Max FoodCourt ",data['FoodCourt'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:34:55.656091Z","iopub.execute_input":"2022-03-10T08:34:55.656926Z","iopub.status.idle":"2022-03-10T08:34:56.931647Z","shell.execute_reply.started":"2022-03-10T08:34:55.656883Z","shell.execute_reply":"2022-03-10T08:34:56.930436Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="FoodCourt", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t FoodCourt")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="FoodCourt", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t FoodCourt")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="FoodCourt", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t FoodCourt")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="FoodCourt", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t FoodCourt")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['FoodCourt'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t FoodCourt")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['FoodCourt'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t FoodCourt")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## ShoppingMall

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:36:11.010024Z","iopub.execute_input":"2022-03-10T08:36:11.010308Z","iopub.status.idle":"2022-03-10T08:36:11.018221Z","shell.execute_reply.started":"2022-03-10T08:36:11.010279Z","shell.execute_reply":"2022-03-10T08:36:11.017181Z"}}

    print("Min ShoppingMall ",data['ShoppingMall'].min())
    print("Mean ShoppingMall ",data['ShoppingMall'].mean())
    print("Max ShoppingMall ",data['ShoppingMall'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:36:38.317972Z","iopub.execute_input":"2022-03-10T08:36:38.318260Z","iopub.status.idle":"2022-03-10T08:36:39.457552Z","shell.execute_reply.started":"2022-03-10T08:36:38.318226Z","shell.execute_reply":"2022-03-10T08:36:39.456746Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="ShoppingMall", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t ShoppingMall")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="ShoppingMall", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t ShoppingMall")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="ShoppingMall", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t ShoppingMall")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="ShoppingMall", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t ShoppingMall")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['ShoppingMall'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t ShoppingMall")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['ShoppingMall'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t ShoppingMall")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## Spa

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:37:50.654979Z","iopub.execute_input":"2022-03-10T08:37:50.655498Z","iopub.status.idle":"2022-03-10T08:37:50.663231Z","shell.execute_reply.started":"2022-03-10T08:37:50.655456Z","shell.execute_reply":"2022-03-10T08:37:50.662249Z"}}
    print("Min Spa ",data['Spa'].min())
    print("Mean Spa ",data['Spa'].mean())
    print("Max Spa ",data['Spa'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:38:17.232375Z","iopub.execute_input":"2022-03-10T08:38:17.233102Z","iopub.status.idle":"2022-03-10T08:38:18.357636Z","shell.execute_reply.started":"2022-03-10T08:38:17.233053Z","shell.execute_reply":"2022-03-10T08:38:18.356235Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="Spa", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t Spa")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="Spa", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t Spa")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="Spa", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t Spa")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="Spa", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t Spa")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['Spa'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t Spa")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['Spa'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t Spa")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## VRDeck

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:39:13.920464Z","iopub.execute_input":"2022-03-10T08:39:13.920783Z","iopub.status.idle":"2022-03-10T08:39:13.928203Z","shell.execute_reply.started":"2022-03-10T08:39:13.920748Z","shell.execute_reply":"2022-03-10T08:39:13.927473Z"}}
    print("Min VRDeck ",data['VRDeck'].min())
    print("Mean VRDeck ",data['VRDeck'].mean())
    print("Max VRDeck ",data['VRDeck'].max())

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:39:37.919434Z","iopub.execute_input":"2022-03-10T08:39:37.919757Z","iopub.status.idle":"2022-03-10T08:39:39.521502Z","shell.execute_reply.started":"2022-03-10T08:39:37.919704Z","shell.execute_reply":"2022-03-10T08:39:39.520649Z"}}
    fig = plt.figure(figsize=(18,10))
    plt.subplot(2,3, 1)
    sns.violinplot(x="HomePlanet",y="VRDeck", hue="Transported", data=data,split=True)
    plt.title("HomePlanet Transported w.r.t VRDeck")
    plt.subplot(2,3, 2)
    sns.violinplot(x="CryoSleep",y="VRDeck", hue="Transported", data=data,split=True)
    plt.title("CryoSleep Transported w.r.t VRDeck")

    plt.subplot(2,3, 3)
    sns.violinplot(x="Destination",y="VRDeck", hue="Transported", data=data,split=True)
    plt.title("Destination Transported w.r.t VRDeck")

    plt.subplot(2,3, 4)
    sns.violinplot(x="VIP",y="VRDeck", hue="Transported", data=data,split=True)
    plt.title("VIP Transported w.r.t VRDeck")

    plt.subplot(2,3, 5)
    data.query('Transported==True')['VRDeck'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(true) w.r.t VRDeck")
    plt.subplot(2,3, 6)
    data.query('Transported==False')['VRDeck'].hist(bins=40,edgecolor='black',grid=False)
    plt.title("Transported(False) w.r.t VRDeck")
    plt.show()
    plt.close()
    
    # %% [markdown]
    # ## Coorelation

    # %% [code] {"execution":{"iopub.status.busy":"2022-03-10T08:40:31.937461Z","iopub.execute_input":"2022-03-10T08:40:31.937779Z","iopub.status.idle":"2022-03-10T08:40:32.476499Z","shell.execute_reply.started":"2022-03-10T08:40:31.937744Z","shell.execute_reply":"2022-03-10T08:40:32.475105Z"}}
    fig = plt.figure(figsize=(10,8))

    sns.heatmap(data.corr(),annot=True)
    plt.title("Coorelation")
    plt.show()
    plt.close()
    