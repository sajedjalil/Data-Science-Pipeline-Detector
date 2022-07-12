import numpy as np 
import pandas as pd
import time 
import logging
from subprocess import check_output
from scipy.stats import variation

class ExpediaHotelRecommendations():
    def __init__(self):
        pass
    
    def variation_analysis(self, data_file, comment=""):
        df = pd.read_csv(data_file, nrows=10)

        #exclude computing the coefficient of variation for all columns of type object  
        print(df.shape) 
        df = df.select_dtypes(exclude=["object"])
        print("shape after exclusion")
        print(df.shape) 
        
        columns = list(df.columns.values)
        results = [] 
        for index, column in enumerate(columns):
            logging.info("computing coefficient of variation for "+column)
            df = pd.read_csv(data_file, usecols=[column])
            var_coeff = variation(df[column])
            print(var_coeff)
            results.append(var_coeff)
            
        variation_results = pd.DataFrame() 
        variation_results["columns"] = columns 
        variation_results["variation"] = results 
        
        variation_results.to_csv("variation_results.csv", sep="|", index=False)  
        
        return
    
    
    
    def descriptive_stats(self, data_file, comment=""):
        descriptive_stats = []
        
        df = pd.read_csv(data_file, nrows=10)
        #exclude computing the coefficient of variation for all columns of type object  
        print(df.shape) 
        df = df.select_dtypes(exclude=["object"])
        print("shape after exclusion")
        print(df.shape) 
        
        columns = list(df.columns.values)
        for index, column in enumerate(columns):
            logging.info("computing descriptive stats for "+column) 
            df = pd.read_csv(data_file, usecols=[column])
            descriptive_stats.append(df.describe().transpose()) 
        
        descriptive_stats = pd.concat(descriptive_stats)
            
        
        logging.info("exporting descriptive stats into a file") 
        descriptive_stats.to_csv("descriptive_stats_"+comment+".csv", sep="|")
        return 
    
    def data_set_distribution_vs_targert_variable(self, data_file, comment=""):
        logging.info("how does the distribution of target variable(hotel clusters) look like?") 
        data = pd.read_csv(data_file, usecols=["hotel_cluster", "is_booking"]) 
        
        print("holel cluster distribution") 
        
        all_data_target_distribution = data["hotel_cluster"].value_counts()
        all_data_target_distribution.to_csv("all_data_target_distribution.csv", sep="|")
        
        print("distribution of clusters on click events")
        click_data_target_distribution = data[data["is_booking"] == 0]["hotel_cluster"].value_counts()
        click_data_target_distribution.to_csv("click_data_target_distribution.csv", sep="|")

        print("distribution of clusters on booking") 
        booking_data_target_distribution = data[data["is_booking"] == 1]["hotel_cluster"].value_counts()
        booking_data_target_distribution.to_csv("booking_data_target_distribution.csv", sep="|")
        
    def clusters_trends_vs_dates(self, data_file, comment=""):
        
        logging.info("trends of clusters over time")
        
        data = pd.read_csv(data_file, usecols=["hotel_cluster", "is_booking","date_time"], parse_dates=["date_time"]) 
        data["year"] = data["date_time"].dt.year  
        
        years = pd.unique(data["year"].values.ravel())

        data = data.groupby("year")
        for year in years:
            #get the dataframe for the current id 
            df = data.get_group(year)
            #export current id's dataframe to a csv file with its name 
            df = df["hotel_cluster"].value_counts()
            df.to_csv("cluster_distribution_per_year_"+str(year)+".csv", sep="|") 
        
        return 
    
        
    
    def exploratory_questions(self, data_file, comment=""):
        
        
        #self.data_set_distribution_vs_targert_variable(data_file, comment)
        
        self.clusters_trends_vs_dates(data_file, comment)    
        
        
        return 
        
    
    def first_peek(self, data, comment=""):
        
        logging.info("peeking into the "+comment+" set") 
        print(data.shape)
        print(data.head())
        
        return
    
    
    def main(self):
    
        ##### Set up python logging format ###### 
        log_format='%(asctime)s %(levelname)s %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)
        logging.info("list of available input files")
        print(check_output(["ls", "../input"]).decode("utf8"))
        train_file = "../input/train.csv"
        test_file = "../input/test.csv" 
        
        ##### load training data ###### 
        logging.info("loading training data")
        #train = pd.read_csv(train_file, delimiter=",", nrows=1000)
        #self.first_peek(train, "train")
        
        #run this to get a report similar to df.describe().transpose() on train data 
        #self.descriptive_stats(train_file, "train") 
        
        #compute the coefficient of variation for each of the column
        #self.variation_analysis(train_file, "train")
        
        #taking a look at the distribution of the target variable 
        self.exploratory_questions(train_file, "train")
        
        logging.info("loading test data")
        #test = pd.read_csv(test_file, delimiter=",")
        #self.first_peek(test, "test")
        
        #run this to get a report similar to df.describe().transpose() on test data 
        #self.descriptive_stats(test_file, "test") 
        
        #compute the coefficient of variation for each of the column
        #self.variation_analysis(test_file, "train")
        
        return


if __name__ == '__main__':
	ehr = ExpediaHotelRecommendations()
	ehr.main()
