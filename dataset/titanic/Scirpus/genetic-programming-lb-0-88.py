import numpy as np
import pandas as pd


def Outputs(data):
    return np.round(1.-(1./(1.+np.exp(-data))))


def GeneticFunction(data):
    return ((np.minimum( ((((0.058823499828577 + data["Sex"]) - np.cos((data["Pclass"] / 2.0))) * 2.0)),  ((0.885868))) * 2.0) +
            np.maximum( ((data["SibSp"] - 2.409090042114258)),  ( -(np.minimum( (data["Sex"]),  (np.sin(data["Parch"]))) * data["Pclass"]))) +
            (0.138462007045746 * ((np.minimum( (data["Sex"]),  (((data["Parch"] / 2.0) / 2.0))) * data["Age"]) - data["Cabin"])) +
            np.minimum( ((np.sin((data["Parch"] * ((data["Fare"] - 0.720430016517639) * 2.0))) * 2.0)),  ((data["SibSp"] / 2.0))) +
            np.maximum( (np.minimum( ( -np.cos(data["Embarked"])),  (0.138462007045746))),  (np.sin(((data["Cabin"] - data["Fare"]) * 2.0)))) +
            -np.minimum( ((((data["Age"] * data["Parch"]) * data["Embarked"]) + data["Parch"])),  (np.sin(data["Pclass"]))) +
            np.minimum( (data["Sex"]),  ((np.sin( -(data["Fare"] * np.cos((data["Fare"] * 1.630429983139038)))) / 2.0))) +
            np.minimum( ((0.230145)),  (np.sin(np.minimum( (((67.0 / 2.0) * np.sin(data["Fare"]))),  (0.31830988618379069))))) +
            np.sin((np.sin(data["Cabin"]) * (np.sin((12.6275)) * np.maximum( (data["Age"]),  (data["Fare"]))))) +
            np.sin(((np.minimum( (data["Fare"]),  ((data["Cabin"] * data["Embarked"]))) / 2.0) *  -data["Fare"])) +
            np.minimum( (((2.675679922103882 * data["SibSp"]) * np.sin(((96) * np.sin(data["Cabin"]))))),  (data["Parch"])) +
            np.sin(np.sin((np.maximum( (np.minimum( (data["Age"]),  (data["Cabin"]))),  ((data["Fare"] * 0.31830988618379069))) * data["Cabin"]))) +
            np.maximum( (np.sin(((12.4148) * (data["Age"] / 2.0)))),  (np.sin((-3.0 * data["Cabin"])))) +
            (np.minimum( (np.sin((((np.sin(((data["Fare"] * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0))),  (data["SibSp"])) / 2.0) +
            ((data["Sex"] - data["SibSp"]) * (np.cos(((data["Embarked"] - 0.730768978595734) + data["Age"])) / 2.0)) +
            ((np.sin(data["Cabin"]) / 2.0) - (np.cos(np.minimum( (data["Age"]),  (data["Embarked"]))) * np.sin(data["Embarked"]))) +
            np.minimum( (0.31830988618379069),  ((data["Sex"] * (2.212120056152344 * (0.720430016517639 - np.sin((data["Age"] * 2.0))))))) +
            (np.minimum( (np.cos(data["Fare"])),  (np.maximum( (np.sin(data["Age"])),  (data["Parch"])))) * np.cos((data["Fare"] / 2.0))) +
            np.sin((data["Parch"] * np.minimum( ((data["Age"] - 1.5707963267948966)),  ((np.cos((data["Pclass"] * 2.0)) / 2.0))))) +
            (data["Parch"] * (np.sin(((data["Fare"] * (0.623655974864960 * data["Age"])) * 2.0)) / 2.0)) +
            (0.31830988618379069 * np.cos(np.maximum( ((0.602940976619720 * data["Fare"])),  ((np.sin(0.720430016517639) * data["Age"]))))) +
            (np.minimum( ((data["SibSp"] / 2.0)),  (np.sin(((data["Pclass"] - data["Fare"]) * data["SibSp"])))) * data["SibSp"]) +
            np.tanh((data["Sex"] * np.sin((5.199999809265137 * np.sin((data["Cabin"] * np.cos(data["Fare"]))))))) +
            (np.minimum( (data["Parch"]),  (data["Sex"])) * np.cos(np.maximum( ((np.cos(data["Parch"]) + data["Age"])),  (3.1415926535897931)))) +
            (np.minimum( (np.tanh(((data["Cabin"] / 2.0) + data["Parch"]))),  ((data["Sex"] + np.cos(data["Age"])))) / 2.0) +
            (np.sin((np.sin(data["Sex"]) * (np.sin((data["Age"] * data["Pclass"])) * data["Pclass"]))) / 2.0) +
            (data["Sex"] * (np.cos(((data["Sex"] + data["Fare"]) * ((8.48635) * (63)))) / 2.0)) +
            np.minimum( (data["Sex"]),  ((np.cos((data["Age"] * np.tanh(np.sin(np.cos(data["Fare"]))))) / 2.0))) +
            (np.tanh(np.tanh( -np.cos((np.maximum( (np.cos(data["Fare"])),  (0.094339601695538)) * data["Age"])))) / 2.0) +
            (np.tanh(np.cos((np.cos(data["Age"]) + (data["Age"] + np.minimum( (data["Fare"]),  (data["Age"])))))) / 2.0) +
            (np.tanh(np.cos((data["Age"] * ((-2.0 + np.sin(data["SibSp"])) + data["Fare"])))) / 2.0) +
            (np.minimum( (((281) - data["Fare"])),  (np.sin((np.maximum( ((176)),  (data["Fare"])) * data["SibSp"])))) * 2.0) +
            np.sin(((np.maximum( (data["Embarked"]),  (data["Age"])) * 2.0) * (((785) * 3.1415926535897931) * data["Age"]))) +
            np.minimum( (data["Sex"]),  (np.sin( -(np.minimum( ((data["Cabin"] / 2.0)),  (data["SibSp"])) * (data["Fare"] / 2.0))))) +
            np.sin(np.sin((data["Cabin"] * (data["Embarked"] + (np.tanh( -data["Age"]) + data["Fare"]))))) +
            (np.cos(np.cos(data["Fare"])) * (np.sin((data["Embarked"] - ((734) * data["Fare"]))) / 2.0)) +
            ((np.minimum( (data["SibSp"]),  (np.cos(data["Fare"]))) * np.cos(data["SibSp"])) * np.sin((data["Age"] / 2.0))) +
            (np.sin((np.sin((data["SibSp"] * np.cos((data["Fare"] * 2.0)))) + (data["Cabin"] * 2.0))) / 2.0) +
            (((data["Sex"] * data["SibSp"]) * np.sin(np.sin( -(data["Fare"] * data["Cabin"])))) * 2.0) +
            (np.sin((data["SibSp"] * ((((5.428569793701172 + 67.0) * 2.0) / 2.0) * data["Age"]))) / 2.0) +
            (data["Pclass"] * (np.sin(((data["Embarked"] * data["Cabin"]) * (data["Age"] - (1.07241)))) / 2.0)) +
            (np.cos((((( -data["SibSp"] + data["Age"]) + data["Parch"]) * data["Embarked"]) / 2.0)) / 2.0) +
            (0.31830988618379069 * np.sin(((data["Age"] * ((data["Embarked"] * np.sin(data["Fare"])) * 2.0)) * 2.0))) +
            ((np.minimum( ((data["Age"] * 0.058823499828577)),  (data["Sex"])) - 0.63661977236758138) * np.tanh(np.sin(data["Pclass"]))) +
            -np.minimum( ((np.cos(((727) * ((data["Fare"] + data["Parch"]) * 2.0))) / 2.0)),  (data["Fare"])) +
            (np.minimum( (np.cos(data["Fare"])),  (data["SibSp"])) * np.minimum( (np.sin(data["Parch"])),  (np.cos((data["Embarked"] * 2.0))))) +
            (np.minimum( (((data["Fare"] / 2.0) - 2.675679922103882)),  (0.138462007045746)) * np.sin((1.5707963267948966 * data["Age"]))) +
            np.minimum( ((0.0821533)),  (((np.sin(data["Fare"]) + data["Embarked"]) - np.cos((data["Age"] * (9.89287)))))))


def MungeData(data):
    # Sex
    data.drop(['Ticket', 'Name'], inplace=True, axis=1)
    data.Sex.fillna('0', inplace=True)
    data.loc[data.Sex != 'male', 'Sex'] = 0
    data.loc[data.Sex == 'male', 'Sex'] = 1
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    # Embarked
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.Embarked.fillna(0, inplace=True)
    data.fillna(-1, inplace=True)
    return data.astype(float)


if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    mungedtrain = MungeData(train)
    trainPredictions = Outputs(GeneticFunction(mungedtrain))

    pdtrain = pd.DataFrame({'PassengerId': mungedtrain.PassengerId.astype(int),
                            'Predicted': trainPredictions.astype(int),
                            'Survived': mungedtrain.Survived.astype(int)})
    pdtrain.to_csv('gptrain.csv', index=False)
    mungedtest = MungeData(test)
    testPredictions = Outputs(GeneticFunction(mungedtest))

    pdtest = pd.DataFrame({'PassengerId': mungedtest.PassengerId.astype(int),
                            'Survived': testPredictions.astype(int)})
    pdtest.to_csv('gptest.csv', index=False)
