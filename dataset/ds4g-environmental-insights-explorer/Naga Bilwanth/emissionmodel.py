import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from emissionglobals import algorithmTypes
from sklearn import metrics
from emissionglobals import modelTrained
import pickle

class emissionModel():
    def __init__(self, appGlobal):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()
        self.outputPath =  self.appGlobal.outputPath
        self.modelTrain = []

    def createModel(self, algorithmType):
        '''
            Read data to be trained and trigger train model function for the provided algorithm type
        '''
        try:
            requiredFields = self.objAppConfig.getAllTrainFields()
            requiredFields.append('emissionFactor')

            for dir in os.listdir(self.outputPath):
                print("Training the model for the sub national region - " + dir)
                if os.path.isdir(self.outputPath + dir):
                    csvEmissionArr = [pd.read_csv(fi) for fi in glob.glob(self.outputPath + dir + "/RasterEmission_*.csv")]
                    csvWeatherArr = [pd.read_csv(fi) for fi in glob.glob(self.outputPath + dir + "/RasterWeather_*.csv")]
                    df_Emission = pd.concat(csvEmissionArr)
                    df_Weather = pd.concat(csvWeatherArr)

                    df_modelData = pd.merge(df_Emission, df_Weather, on=['Date',dir])
                    df_modelData = df_modelData[requiredFields]

                    x = df_modelData.iloc[:,:-1]
                    y = df_modelData.iloc[:, -1]

                    self.runAlgorithm(x, y, algorithmType, self.outputPath + dir, df_Emission)

            finalModel = None

            for md in self.modelTrain:
                if finalModel == None:
                    finalModel = md
                else:
                    if(finalModel.modelTrainCoef < md.modelTrainCoef and \
                        finalModel.modelTestCoef < md.modelTestCoef and finalModel.modelRMSE > md.modelRMSE):
                        finalModel = md

            modelPath = os.path.dirname(os.path.dirname(self.appGlobal.outputPath))            
            pickle.dump(finalModel.modelObj, open(modelPath + "\\emissionModel_trained.h5", 'wb'))

        except Exception as e:
            print(e)
            print("Failed at emissionmodel.py - createModel")
            raise
        
    def runAlgorithm(self, x, y, algorithmType, outputPath, xdummy):
        '''
           Traing the model for provided data and algorithm type
        '''
        try:
            x_train, x_test,y_train, y_test = train_test_split(x, y)
            mdl = None
            mt = modelTrained()
            
            if algorithmTypes.randomForestRegression == algorithmType:
                from sklearn.ensemble import RandomForestRegressor
                mt.modelObj = RandomForestRegressor(n_estimators=100)            

            if algorithmTypes.linearRegression == algorithmType:            
                from sklearn.linear_model import LinearRegression
                mt.modelObj = LinearRegression()

            if algorithmTypes.xgboost == algorithmType:            
                import xgboost as xgb
                '''
                params={
                    "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    "max_depth" : [3, 4, 5, 6, 8, 10, 12, 15],
                    "min_child_weight" : [1,3,5,7],
                    "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
                    "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
                    }
                    
                mdl = self.deriveHyperParameters(x, y, params, mdl) #xgb.XGBRegressor()'''

                mt.modelObj = xgb.XGBRegressor()
                
            #Train the model
            mt.modelObj.fit(x_train, y_train)

            # calculating the Root mean square error
            mt.modelTrainCoef = mt.modelObj.score(x_train, y_train)
            mt.modelTestCoef = mt.modelObj.score(x_test, y_test)
            print('Co-efficient of determination -- on train set - ', mt.modelTrainCoef)
            print('Co-efficient of determination -- on test set - ', mt.modelTestCoef)

            #mode evaluation
            y_prediction = mt.modelObj.predict(x_test)

            # calculating the Root mean square error
            mt.modelMSE = metrics.mean_squared_error(y_test, y_prediction)
            mt.modelRMSE = np.sqrt(mt.modelMSE)
            
            print('MSE - ', mt.modelMSE)
            print('RMSE - ', mt.modelRMSE )

            # Plot test and prediction data
            sns.scatterplot(y_test, y_prediction)
            plt.xlabel = "Prediction data"
            plt.ylabel = "Test Data"

            self.modelTrain.append(mt)
            
            plt.savefig(outputPath + "/"+ algorithmType+"_train_coeff"+str(mt.modelTrainCoef)+"_test_coeff_"+str(mt.modelTestCoef)+".png")
        except Exception as e:
            print(e)
            print("Failed at emissionmodel.py - runAlgorithm")
            raise

    def deriveHyperParameters(self,x, y, params, regressor):
        try:
            random_search = RandomizedSearchCV(regressor, param_distributions=params,n_iter=5, \
                scoring="r2", n_jobs=-1, verbose=3)

            random_search.fit(x, y)

            return random_search.best_estimator_
        except Exception as e:
            print(e)
            print("Failed at emissionmodel.py - deriveHyperParameters")
            raise

    def predictEmissionFactor(self, date, subregion):
        """
            Predict the emission factor for provided weather and emission data
        """
        try:            
            modelpath = self.config["configList"]["emissionModel"]
            requiredFields = self.objAppConfig.getAllTrainFields()        

            dir = subregion + "_" + date
            print("Predicting the emission factor - " + dir)

            if os.path.isdir(self.outputPath + dir):
                csvEmissionArr = [pd.read_csv(fi) for fi in glob.glob(self.outputPath + dir + "/RasterEmission_*.csv")]
                csvWeatherArr = [pd.read_csv(fi) for fi in glob.glob(self.outputPath + dir + "/RasterWeather_*.csv")]
                df_Emission = pd.concat(csvEmissionArr)
                df_Weather = pd.concat(csvWeatherArr)

                df_modelData = pd.merge(df_Emission, df_Weather, on=subregion)
                df_modelData = df_modelData[requiredFields]

                x_predict = df_modelData.iloc[:,:]

                # load the model from disk
                loaded_model = pickle.load(open(modelpath, 'rb'))
                result = loaded_model.predict(x_predict)

                df_modelData["EmissionFactor"] = result

                df_modelData.to_csv(self.outputPath + dir + "/" + "emission_predicted.csv")

                print("Output saved at this location - ", self.outputPath + dir + "/" + "emission_predicted.csv")
                print("Predicting the emission factor is done ..")

        except Exception as e:
            print(e)
            print("Failed at emissionmodel.py - predictEmissionFactor")
            raise