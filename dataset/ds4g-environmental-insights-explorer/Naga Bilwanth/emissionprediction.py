import sys, os
import json
from emissionanalyzer import emissionFactorAnalyzer
from emissionweather import weatherAnalyzer
from emissionconfig import appConfig
from emissionglobals import appGlobal
from emissionplot import plotGraph
from emissionspatial import emissionSpatialLayer
import shutil
import pandas as pd
from emissionmodel import emissionModel
from emissionglobals import algorithmTypes
from emissionspatial import geoLocation

exePath = "/kaggle/input/"

def setGlobal(args):
    global objConfig
    objConfig = appConfig(exePath + "ds4gconfiguration/config_predict_kaggle.json")
    outputPath = "/kaggle/working/output_predict/"

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if not os.path.exists(outputPath+"/"+args[2]+"_"+args[1]):
        os.makedirs(outputPath+"/"+args[2]+"_"+args[1])
    else:        
        shutil.rmtree(outputPath+"/"+args[2]+"_"+args[1], ignore_errors=True)
        
        if not os.path.exists(outputPath+"/"+args[2]+"_"+args[1]):
            os.makedirs(outputPath+"/"+args[2]+"_"+args[1])
    
    global aGlobal
    aGlobal = appGlobal(objConfig, outputPath)

    global layerList
    layerList = objConfig.getLayerList()

def convertCSVToshapefile():
    generatorObj = objConfig.getGenerationConfigObj("Generator")

    requiredColumns = []
    for key in generatorObj["required_columns"]:
        requiredColumns.append(generatorObj["required_columns"][key])

    gl = geoLocation(generatorObj, requiredColumns)

    gl.getGeoLocationAsDataframe()

def main(args):
    try:
        setGlobal(args)

        for layer in layerList:
            if(layer == args[2]): 
                print('processing layer ' + layer)
                spl = emissionSpatialLayer(aGlobal, layer)
                dictSubRegions = spl.getSpatialLayerNPArr()
                       
                for subRegion in dictSubRegions:                
                    dfGeoLocation = spl.getGeneratorXYLocation(subRegion)
                    print('processing '+ layer + " "+ subRegion)
                    
                    if not dfGeoLocation.empty:
                        rpE = emissionFactorAnalyzer(aGlobal, "RasterEmission")
                        rpE.generateEF_SubRegion(dictSubRegions[subRegion], dfGeoLocation, subRegion, layer, args[1])

                        rpW = weatherAnalyzer(aGlobal, "RasterWeather")
                        rpW.getWeather_subRegion(dictSubRegions[subRegion], subRegion, layer, args[1])
                    
            
        print("create Model to train machine with emission factor")
        em = emissionModel(aGlobal)
        em.predictEmissionFactor(args[1], args[2])

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    args=["","07-04-2018","County"]
            
    main(args)