import sys, os, json
from datetime import datetime

class appGlobal():
    def __init__(self, objAppConfig, outputPath):
        self.objAppConfig = objAppConfig
        self.outputPath = outputPath
        self.ConversionToKiloMeter = 6371
        self.ConversionToMiles = 3959

class appBandInfo():
    def __init__(self, aliasName, color, description, plot, linewidth, train):
        self.aliasName = aliasName
        self.color = color
        self.description = description
        self.plot = plot
        self.linewidth = linewidth
        self.train = train

class algorithmTypes():    
    randomForestRegression = "RandomForestRegression"
    linearRegression = "LinearRegression"
    ridgeRegression = "RidgeRegression"
    lassoRegression = "LassoRegression"
    xgboost = "xgboost"
    ArtificalNeuralNetwork = "ArtificalNeuralNetwork"

class modelTrained():
    modelObj = None
    modelMSE = -1
    modelRMSE = -1
    modelTrainCoef = -1
    modelTestCoef = -1

class appCommon():
    def __init__(self):
        pass

    def GetJsonFromFile(self, filePath):
        """
            Read Json file by skiping comments
        """
        try:
            contents = ""
            fh = open(filePath)
            for line in fh:
                cleanedLine = line.split("//", 1)[0]
                if len(cleanedLine) > 0 and line.endswith("\n") and "\n" not in cleanedLine:
                    cleanedLine += "\n"
                contents += cleanedLine
            fh.close
            while "/*" in contents:
                preComment, postComment = contents.split("/*", 1)
                contents = preComment + postComment.split("*/", 1)[1]
            return contents
        except Exception as e:
            print(e)
            print("Failed at emissionglobals.py - GetJsonFromFile")
            raise

    def timer(self, startTime=None):
        try:
            if not startTime:
                startTime = datetime.now()
                return startTime
            elif startTime:
                thour, temp_sec = divmod((datetime.now() - startTime).total_seconds(), 3600)
                tmin, tsec = divmod(temp_sec, 60)
                print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        except Exception as e:
            print(e)
            print("Failed at emissionglobals.py - timer")
            raise
            
