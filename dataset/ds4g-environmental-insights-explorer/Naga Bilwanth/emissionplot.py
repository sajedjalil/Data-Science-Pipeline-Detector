import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import glob 

class plotGraph():
    def __init__(self, appGlobal):
        self.appGlobal = appGlobal
        self.objAppConfig = appGlobal.objAppConfig
        self.config = self.objAppConfig.getConfig()
        self.weatherAliasList = self.objAppConfig.getBandAliasAndDescription("RasterWeather")
        self.emissionAliasList = self.objAppConfig.getBandAliasAndDescription("RasterEmission")

    def plotSelectedRegionEmissions(self,layerName, subRegionName, plotBy, plot, xLabel, yLabel):
        '''
            Plot selected region
        '''
        try:
            df_Weather = pd.read_csv(self.appGlobal.outputPath + layerName + "/RasterWeather_"+subRegionName+".csv")
            
            if plotBy == "Month":
                df_Weather = self.prepareDataToPlot("RasterWeather",self.weatherAliasList, df_Weather)

            df_Emission = pd.read_csv(self.appGlobal.outputPath +layerName+ "/RasterEmission_"+subRegionName+".csv")

            if plotBy == "Month":
                df_Emission = self.prepareDataToPlot("RasterEmission",self.emissionAliasList, df_Emission)
            
            plotpath = self.appGlobal.outputPath +  layerName + "/Plot/" + plot + "/"
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)

            ss = StandardScaler()
            df_Merge = df_Weather.merge(df_Emission, left_on=plotBy, right_on=plotBy)

            df_Merge["date_month"] = pd.to_datetime(df_Merge.index, dayfirst=False)
            df_Merge = df_Merge.sort_values("date_month")
            df_Merge.drop(columns=["date_month"], inplace=True)

            if(plotBy == 'Date'):
                df_date = df_Merge[["Date","Month_x","Year_x",layerName+ "_x"]]
                df_date.rename(columns={"Month_x":"Month","Year_x":"Year",layerName+ "_x": layerName})
                df_NoDate = df_Merge.drop(columns=["Date","Month_x","Year_x",layerName + "_x","Month_y","Year_y",layerName + "_y"])

                df_NoDate_scaled = pd.DataFrame(ss.fit_transform(df_NoDate), columns=df_NoDate.columns)            
                df_Merge_scaled = df_NoDate_scaled.merge(df_date, left_index=True, right_index=True)

            if(plotBy == 'Month'):   
                df_Merge["Month"] = df_Merge.index    
                df_date = df_Merge[["Month"]]
                #df_date.set_index = df_Merge["Month"]
                df_NoDate = df_Merge.drop(columns=["Month"])
                #df_NoDate.set_index = df_Merge["Month"]

                df_NoDate_scaled = pd.DataFrame(ss.fit_transform(df_NoDate), columns=df_NoDate.columns)   
                df_NoDate_scaled.set_index(df_Merge["Month"], inplace=True)
                df_Merge_scaled = df_NoDate_scaled.merge(df_date, left_index=True, right_index=True)

            fig = plt.figure(figsize=(12, 8))
            plt.xticks(rotation=90)

            if(plot == "WeatherAndEmission"):
                for binfo in self.weatherAliasList:  
                    if binfo.plot:                      
                        plt.plot( plotBy, binfo.aliasName, data=df_Merge_scaled, marker='', color=binfo.color, linewidth=binfo.linewidth, label=binfo.description)

            if(plot == "WeatherAndEmission" or plot == "Emission"):
                for binfo in self.emissionAliasList:
                    if binfo.plot:
                        plt.plot( plotBy, binfo.aliasName, data=df_Merge_scaled, marker='', color=binfo.color, linewidth=binfo.linewidth, label=binfo.description)

            plt.plot(plotBy, 'emissionFactor', data=df_Merge_scaled, marker='o', color='red', linewidth=2, label='Emission Factor')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=3, fancybox=True, shadow=True)

            plt.xlabel(xLabel)
            plt.ylabel(yLabel)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.autoscale(axis='x', tight=True)
            plt.savefig(plotpath + subRegionName + ".png",bbox_inches = 'tight',pad_inches = 0)
            plt.close(fig)
        except Exception as e:
            print(e)
            print("Failed at emissionplot.py - plotSelectedRegionEmissions")
            raise
    
    def plotEFRegionComparison(self,layerName):
        '''
            Plot comparsion between Emission factor across region
        '''
        try:
            path = self.appGlobal.outputPath + layerName + "/RasterEmission_*.csv"        
            list_dfEmission = [pd.read_csv(fname) for fname in glob.glob(path)]

            df_Emission = pd.concat(list_dfEmission)
            
            df_Emission["date_month"] = pd.to_datetime(df_Emission.index, dayfirst=False)
            df_Emission = df_Emission.sort_values("date_month")
            df_Emission.drop(columns=["date_month"], inplace=True)

            fig = plt.figure(figsize=(10, 6))

            sns.barplot(x="emissionFactor",
                y=layerName,
                data=df_Emission, ci=None)

            plt.xlabel("Emission Factor")
            plt.ylabel(layerName)
            #plt.show()

            plotpath = self.appGlobal.outputPath +  layerName + "/Plot/" #+ plot + "/"

            if not os.path.exists(plotpath):
                os.makedirs(plotpath)

            plt.savefig(plotpath + layerName + ".png", bbox_inches = 'tight',pad_inches = 0)
            plt.close(fig)
        except Exception as e:
            print(e)
            print("Failed at emissionplot.py - plotEFRegionComparison")
            raise
    
    def prepareDataToPlot(self,dfType, bandDesc, dFrame):
        try:
            concatType = self.objAppConfig.getConcatType(dfType)
            fields = self.objAppConfig.getAllPlotFields(dfType)

            if concatType == "sum":
                fields.append('emissionFactor')
                dfMonth = dFrame.groupby('Month')[fields].apply(lambda x : x.astype(float).sum())

            if concatType == "average":
                dfMonth = dFrame.groupby('Month')[fields].apply(lambda x : x.astype(float).mean())

            return dfMonth
        except Exception as e:
            print(e)
            print("Failed at emissionplot.py - prepareDataToPlot")
            raise

    def getListofMonth(self, dFrameEmission, dFrameWeather ):
        try:
            df_Merge = dFrameEmission.merge(dFrameWeather, on="Month")
            month_list = df_Merge.groupby('Month')["Month"].unique()
            return month_list
        except Exception as e:
            print(e)
            print("Failed at emissionplot.py - getListofMonth")
            raise

    def plotSelectedRegionEmissionsMonthToMonth(self,layerName, subRegionName, plotBy, plot, xLabel, yLabel):
        '''
        '''
        try:
            df_Weather = pd.read_csv(self.appGlobal.outputPath + layerName + "/RasterWeather_"+subRegionName+".csv")
                    
            df_Emission = pd.read_csv(self.appGlobal.outputPath +layerName+ "/RasterEmission_"+subRegionName+".csv")
            
            month_list = self.getListofMonth(df_Emission, df_Weather)

            for montharr in month_list:
                month = montharr[0]
                df_weather_month = df_Weather[df_Weather["Month"] == month]
                df_Emission_month = df_Emission[df_Emission["Month"] == month]

                plotpath = self.appGlobal.outputPath +  layerName + "/Plot/" + plot + "/month/"
                if not os.path.exists(plotpath):
                    os.makedirs(plotpath)

                ss = StandardScaler()
                df_Merge = df_weather_month.merge(df_Emission_month, left_on=plotBy, right_on=plotBy)

                df_Merge["date_sort"] = pd.to_datetime(df_Merge["Date"], dayfirst=False)   
                df_Merge = df_Merge.sort_values("date_sort")
                df_Merge.drop(columns=["date_sort"], inplace=True)             
                
                df_date = df_Merge[["Date","Month_x","Year_x",layerName+ "_x"]]
                df_date.rename(columns={"Month_x":"Month","Year_x":"Year",layerName+ "_x": layerName})
                df_NoDate = df_Merge.drop(columns=["Date","Month_x","Year_x",layerName + "_x","Month_y","Year_y",layerName + "_y"])

                df_NoDate_scaled = pd.DataFrame(ss.fit_transform(df_NoDate), columns=df_NoDate.columns)            
                df_Merge_scaled = df_NoDate_scaled.merge(df_date, left_index=True, right_index=True)
                
                fig = plt.figure(figsize=(12, 8))
                plt.xticks(rotation=90)

                if(plot == "WeatherAndEmission"):
                    for binfo in self.weatherAliasList:  
                        if binfo.plot:                      
                            plt.plot( plotBy, binfo.aliasName, data=df_Merge_scaled, marker='', color=binfo.color, linewidth=binfo.linewidth, label=binfo.description)

                if(plot == "WeatherAndEmission" or plot == "Emission"):
                    for binfo in self.emissionAliasList:
                        if binfo.plot:
                            plt.plot( plotBy, binfo.aliasName, data=df_Merge_scaled, marker='', color=binfo.color, linewidth=binfo.linewidth, label=binfo.description)

                plt.plot(plotBy, 'emissionFactor', data=df_Merge_scaled, marker='o', color='red', linewidth=2, label='Emission Factor')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                ncol=3, fancybox=True, shadow=True)

                plt.xlabel(xLabel)
                plt.ylabel(yLabel)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.autoscale(axis='x', tight=True)
                plt.savefig(plotpath + subRegionName +"_"+month + ".png",bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)

        except Exception as e:
            print(e)
            print("Failed at emissionplot.py - plotSelectedRegionEmissionsMonthToMonth")
            raise