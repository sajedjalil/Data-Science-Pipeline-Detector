import pandas,numpy
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier\
     ,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import preprocessing

path_file='../input/application_train.csv'
df = pandas.read_csv(path_file)

dynamic_array=[]
for item in df.columns:
    if df[item].dtype =='object':
        dynamic_array.append(item)

for item1 in dynamic_array:
    dynamic_array2=list(set(df[item1]))
    new=[]
    for item2 in df[item1]:
        for item3 in dynamic_array2:
            if  item2==item3 or item2 is item3:
                new.append(dynamic_array2.index(item3))
    df[item1]=new

something=df.columns[df.isna().any()].tolist()
for item in something:
    df[item].fillna(0,inplace=True)

data_frame_target=df['TARGET']
del df['TARGET']

feature=df.values
pca_dims = PCA()
pca_dims.fit(feature)
cumsum = numpy.cumsum(pca_dims.explained_variance_ratio_)
d = numpy.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=d)
feature = pca.fit_transform(feature)
feature=preprocessing.scale(feature)
target=data_frame_target.values

feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target, test_size=0.1, random_state=42)

classifier_dict={'KNeighborsClassifier':KNeighborsClassifier,'LinearSVC':LinearSVC,'SVC':SVC,\
'RandomForestClassifier':RandomForestClassifier,'ExtraTreesClassifier':ExtraTreesClassifier,\
'AdaBoostClassifier':AdaBoostClassifier,'GradientBoostingClassifier':GradientBoostingClassifier,\
'DecisionTreeClassifier':DecisionTreeClassifier,'MLPClassifier':MLPClassifier}

accuracy_array=[]

for key,value in classifier_dict.items():
    accuracy_array.append(key)
    classifier=value()
    classifier.fit(feature_train, target_train)
    predict=classifier.predict(feature_test)
    accuracy=accuracy_score(target_test,predict)
    accuracy_array.append(accuracy)
    accuracy_array.append(' ')
    
numpy.savetxt('accurancy.txt',accuracy_array,fmt='%s')
