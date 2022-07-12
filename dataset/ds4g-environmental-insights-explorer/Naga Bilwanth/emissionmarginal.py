import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import glob
from emissiondataset import generatorDataset
from itertools import groupby

class marginalemissionfactor:
    """
        This class used to calculate Marginal Emission Factor
    """
    def __init__(self, appGlobal):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()
        self.generatorObj = self.objAppConfig.getGenerationConfigObj("Generator")
        self.getBandAlias = self.objAppConfig.getBandAlias("RasterEmission")

    def calculateMarginalEmissions(self, layerName):
        """
            Calculate Marginal Emission Factor and create plot with comparison 
            marginal emission factor, estimated capacity of plant and primary fuel
        """
        try:
            filepath = self.appGlobal.outputPath + layerName + "/RasterEmission_*.csv"
            fileList = glob.glob(filepath)
            self.getBandAlias.append(layerName)
            self.getBandAlias.append("Date")
            self.getBandAlias.append("Month")

            df_Emission_arr = [pd.read_csv(file, usecols=self.getBandAlias) for file in fileList]
            df_emission = pd.concat(df_Emission_arr)

            requiredColumns = []
            for key in self.generatorObj["margin_emission_columns"]:
                requiredColumns.append(self.generatorObj["margin_emission_columns"][key])

            ge = generatorDataset(self.appGlobal, "Generator")
            df_gppd = pd.read_csv(self.generatorObj["data_path"], usecols=requiredColumns)
            df_gppd = ge.fix_estimated_generation(df_gppd)

            df_merge = pd.merge(df_emission, df_gppd, left_on=layerName, right_on="name")
            
            # converting annual estimated power generation to individual day
            df_merge["estimated_generation_gwh_per_Day"] = df_merge["estimated_generation_gwh"]/365

            aliasNames = self.objAppConfig.getEmissionAliasNames("RasterEmission")

            cnt = 0
            # sum fields configured as emission = true to calculate emission Factor
            for alias in aliasNames:
                if cnt == 0:
                    df_merge["emissionSum"] = df_merge[alias]
                    cnt = 1
                else:
                    df_merge["emissionSum"] = df_merge["emissionSum"] + df_merge[alias]

            #df_merge["marginEmissionFactor"] = df_merge["emissionSum"] / df_merge["estimated_generation_gwh_per_Day"]

            df_plot = self.prepareDataToPlot(df_merge)
            
            fig = plt.figure(figsize=(22, 8))            
            #plt.xticks(rotation=90)
            #ax = df_plot.groupby(['Month','primary_fuel']).size().unstack().plot(kind='bar',y='estimated_generation_gwh_per_Day',stacked=True)

            ax = df_plot.plot(kind='bar',x='primary_fuel',y='estimated_generation_gwh_per_Day',color='red') #df['perc'].plot(kind="bar", alpha=0.7)

            ax2 = ax.twinx()
            ax2.plot(ax.get_xticks(),df_plot['marginEmissionFactor'],marker='o', c='navy', linewidth=4)
            
            ax.set_xticklabels(df_plot['primary_fuel'] + "^" + df_plot['Month'])
            ax2.set_xticklabels('')
            ax.set_ylabel('Estimated Power generation in GWH Per Month')
            ax2.set_ylabel('Marginal EMission Factor')
            self.plotLabel(ax, ax2, df_plot)
           
            #fig.tight_layout()  # otherwise the right y-label is slightly clipped

            #plt.show()
            fig = plt.gcf()
            fig.set_figheight(8)
            fig.set_figwidth(20)
            fig.savefig(self.appGlobal.outputPath + layerName + "/" + layerName + "_MEF.png") #,bbox_inches = 'tight',pad_inches = 0
            plt.close(fig)

        except Exception as e:
            print(e)
            print("Failed at emissionmarginal.py - calculateMarginalEmissions")
            raise

    def prepareDataToPlot(self, dFrame):
        """
            prepare data frame with margin emission factor
        """
        try:
            fields = ["emissionSum", "estimated_generation_gwh_per_Day"]
            dfArr_fuel_list = []

            for index, row_fuel in dFrame.groupby(['primary_fuel','Month']):
                df = row_fuel.groupby('Month')[fields].apply(lambda x : x.astype(float).sum())
                
                df["marginEmissionFactor"] = df["emissionSum"] / df["estimated_generation_gwh_per_Day"]            
                
                df["primary_fuel"] = index[0]
                df["Month"] = index[1]
                dfArr_fuel_list.append(df)
                            
            df_plot = pd.concat(dfArr_fuel_list)
            return df_plot
        except Exception as e:
            print(e)
            print("Failed at emissionmarginal.py - prepareDataToPlot")
            return None

    def plotLabel(self, ax, ax2, df):
        """
            Arrange labels in plot
        """
        try:
            preLbl = None
            currLbl = None
            cnt = 0
            labelPos = 0  
            labelUpdated = []
            scale = 1./df.index.size
            for label in ax.get_xticklabels():
                currLbl = label._text.split("^")[0]            
                labelUpdated.append("")
                labelPos += 1
                xpos = labelPos * scale - 0.005
                mlabel = label._text.split("^")[1][0:3] + "" + label._text.split("^")[1][-2:] 
                ax.text(xpos, -0.085, mlabel, ha='center', transform=ax.transAxes, rotation=90)
                if currLbl == preLbl or preLbl == None:
                    cnt += 1
                    preLbl = currLbl
                else:
                    pos = int(abs(cnt/2))                
                    labelUpdated[labelPos - pos] = preLbl
                            
                    preLbl = currLbl
                    cnt= 0

            labelUpdated[labelPos - pos] = preLbl        
            ax.set_xticklabels(labelUpdated,{'rotation':0, 'x':-0.09, 'y':-0.09})
        except Exception as e:
            print(e)
            print("Failed at emissionmarginal.py - plotLabel")
