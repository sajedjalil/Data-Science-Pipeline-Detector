import sys, os, json
from emissionglobals import appCommon
from emissionglobals import appBandInfo

class appConfig():
    """
        Provide functionality to read app configuration file
    """

    def __init__(self, configFile):
        self.configFile = configFile
        self.config = self.getConfig()
        

    def getConfig(self):
        """
            Get Config.json content
        """
        try:
            appC = appCommon()
            content = appC.GetJsonFromFile(self.configFile)
                
            #with open(self.configFile) as configFile:
            self.config = json.loads(content)

            return self.config
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getConfig")
            raise

    def getBandCount(self, rasterType):
        """
            Get Band count considered for Analysis purpose
        """
        try:
            cnt = 0
            for cElements in self.config:      
                #print(cElements)              
                if self.config[cElements]["type"] == rasterType:                                
                    for bandElements in self.config[cElements]["bandsConsidered"]:                    
                        cnt += 1

            return cnt
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getBandCount")
            raise

    def getBandDescriptions(self, rasterType):    
        """
            Get Band description for selected raster type
        """   
        try:    
            bandDescArr=[]
            for cElements in self.config:      
                #print(cElements)              
                if self.config[cElements]["type"] == rasterType:                                
                    for bandElements in self.config[cElements]["bandsConsidered"]:                    
                        bandDescArr.append(bandElements['description'])

            return bandDescArr
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getBandDescriptions")
            raise

    def getEmissionAliasNames(self, rasterType):  
        """
            Get Emission alias for selected raster type
        """   
        try:   
            bandEmission=[]
            for cElements in self.config:      
                #print(cElements)              
                if self.config[cElements]["type"] == rasterType:                                
                    for bandElements in self.config[cElements]["bandsConsidered"]:   
                        
                        if bandElements["emission"]:
                            bandEmission.append(bandElements['aliasName'])

            return bandEmission
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getEmissionAliasNames")
            raise

    def getBandAlias(self, rasterType):  
        """
            Get Band Alias selected raster type
        """   
        try:   
            bandAlias=[]
            for cElements in self.config:      
                #print(cElements)              
                if self.config[cElements]["type"] == rasterType:                                
                    for bandElements in self.config[cElements]["bandsConsidered"]:                    
                        bandAlias.append(bandElements['aliasName'])

            return bandAlias
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getBandAlias")
            raise

    def getBandAliasAndDescription(self, rasterType):    
        """
            Get Band Alias and Description for selected raster type
        """
        try:    
            bandAliasDict = []
            for cElements in self.config:
                if self.config[cElements]["type"] == rasterType:
                    for bandElements in self.config[cElements]["bandsConsidered"]:   
                        aliasName = bandElements['aliasName']
                        color = bandElements['color']
                        description = bandElements['description']
                        plot = bandElements['plot']
                        linewidth = bandElements['linewidth']
                        train = bandElements['train']

                        bInfo = appBandInfo(aliasName, color, description, plot, linewidth, train)
                        bandAliasDict.append(bInfo)

            return bandAliasDict
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getBandAliasAndDescription")
            raise

    def getGenerationConfigObj(self, type):
        """
            Get config object for provided type
        """
        try:
            generateObj = []
            
            for cElements in self.config:      
                if self.config[cElements]["type"] == type:
                    generateObj = self.config[cElements]

            return generateObj
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getGenerationConfigObj")
            raise
    
    def getlayerInfo(self, layerName):
        """
            Get layer info listed in Spatial node
        """   
        try:     
            referenceRaster = None

            for cElements in self.config:      
                if self.config[cElements]["type"] == "spatial":
                    layerArr = self.config[cElements]["layers"]
                    referenceRaster = self.config[cElements]["rasterReference"]
                    projection = self.config[cElements]["projection"]

                    for layer in layerArr:
                        if layer["name"] == layerName:
                            return layer, referenceRaster, projection

            return None, referenceRaster
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getlayerInfo")
            raise

    def getLayerList(self):
        """
            Get Layers specified in Spatial node in config
        """
        try:
        
            layerList = []

            for cElements in self.config:
                if self.config[cElements]["type"] == "spatial":
                    layerArr = self.config[cElements]["layers"]
                    
                    for layer in layerArr:
                        layerList.append(layer["name"])

            return layerList
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getLayerList")
            raise

    def getAllTrainFields(self):
        """
            Get raster bands that need to be considered for Training the Model
        """
        try:
            trainFields = []
            for cElements in self.config:
                if self.config[cElements]["type"] == "RasterEmission" or \
                    self.config[cElements]["type"] == "RasterWeather":

                    for bandElements in self.config[cElements]["bandsConsidered"]:                   
                        if(bandElements['train']):
                            trainFields.append(bandElements['aliasName'])

            return trainFields
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getAllTrainFields")
            raise

    def getConcatType(self, rasterType):
        """
            Get concat type that need to performed on dataset
        """
        try:
            for cElements in self.config:
                    if self.config[cElements]["type"] == rasterType:
                        return self.config[cElements]["concatType"]
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getConcatType")
            raise        

    def getAllPlotFields(self, rasterType):
        """
            Get all fields that are specified as true in config
        """
        try:
            plotFields = []
            for cElements in self.config:
                if self.config[cElements]["type"] == rasterType:

                    for bandElements in self.config[cElements]["bandsConsidered"]:                   
                        if(bandElements['plot']):
                            plotFields.append(bandElements['aliasName'])

            return plotFields
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getAllPlotFields")
            raise

    def getLayerType(self, layerName):
        """
            Get Layer Type from the spatial node in config
        """
        try:
            for cElements in self.config:
                    if self.config[cElements]["type"] == "spatial":
                        for layer in self.config[cElements]["layers"]:
                            if layer["name"] == layerName:
                                return layer["type"]
                    
            return None
        except Exception as e:
            print(e)
            print("Failed at emissionconfig.py - getLayerType")
            raise