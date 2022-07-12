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
from emissionmarginal import marginalemissionfactor
from emissionspatial import geoLocation

exePath = "/kaggle/input/" #os.path.dirname(__file__)

def setGlobal():
    global objConfig
    objConfig = appConfig(exePath + "ds4gconfiguration/config_kaggle.json")
    outputPath = "/kaggle/working/output/"

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    else:        
        shutil.rmtree(outputPath, ignore_errors=True)
        
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
    
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

def main():
    try:
        setGlobal()

        convertCSVToshapefile()

        for layer in layerList:
            print('processing layer ' + layer)
            spl = emissionSpatialLayer(aGlobal, layer)
            dictSubRegions = spl.getSpatialLayerNPArr()
            
            for subRegion in dictSubRegions:
                dfGeoLocation = spl.getGeneratorXYLocation(subRegion)
                print('processing '+ layer + " "+ subRegion)
                
                if not dfGeoLocation.empty:
                    rpE = emissionFactorAnalyzer(aGlobal, "RasterEmission")
                    rpE.generateEF_SubRegion(dictSubRegions[subRegion], dfGeoLocation, subRegion, layer)

                    rpW = weatherAnalyzer(aGlobal, "RasterWeather")
                    rpW.getWeather_subRegion(dictSubRegions[subRegion], subRegion, layer)
                    
                    g = plotGraph(aGlobal)
                    
                    print("plot analysis report by compare Weather And emission factor and Nox emissions, for all data")
                    g.plotSelectedRegionEmissions(layer, subRegion, "Month", "WeatherAndEmission","","Emission Factor Vs Weather")

                    print("plot analysis report by compare emission factor and Nox emissions, for all data")
                    g.plotSelectedRegionEmissions(layer, subRegion, "Month", "Emission","Time Interval","Emission Factor Vs No2 Emission")

                    print("plot analysis report by compare emission factor and Nox emissions, Month by Month")
                    g.plotSelectedRegionEmissionsMonthToMonth(layer, subRegion, "Date", "Emission","Time Interval","Emission Factor Vs No2 Emission")
                    
                    print("plot analysis report to compare Weather And emission factor and Nox emissions, Month by Month")
                    g.plotSelectedRegionEmissionsMonthToMonth(layer, subRegion, "Date", "WeatherAndEmission","Time Interval","Emission Factor Vs No2 Emission")
            
            print("plot Emission factor comparison across sub regions")
            g = plotGraph(aGlobal)
            g.plotEFRegionComparison(layer)

            print("calculate Marginal emission factor for power plants")
            if objConfig.getLayerType(layer) == "powerplant_subregion":
                me = marginalemissionfactor(aGlobal)
                me.calculateMarginalEmissions(layer)

        print("create Model to train machine with emission factor")
        em = emissionModel(aGlobal)
        em.createModel(algorithmTypes.randomForestRegression)

    except Exception as ex:
        print(ex)

if __name__ == "__main__":
    main()