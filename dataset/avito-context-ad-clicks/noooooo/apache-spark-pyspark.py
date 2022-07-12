#Following script will only be useful if anybody is interested in trying the same in Apache Spark Environment
#Script is based on Apache Spark dataframe & Spark SQL
#can be used for basic operations like filter , groupby and others

#Loading each tsv as dataframe


df_ads = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/AdsInfo.tsv")
df_ads.registerTempTable("ads")
