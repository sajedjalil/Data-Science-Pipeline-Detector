#Following script will only be useful if anybody is interested in trying the same in Apache Spark Environment
#Script is based on Apache Spark dataframe & Spark SQL
#can be used for basic operations like filter , groupby and others

#Loading each tsv as dataframe
df_loc = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/Location.tsv")
#register this to access the dataframe using SQL
df_loc.registerTempTable("loc")

df_cat = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/Category.tsv")
df_cat.registerTempTable("cat")

df_ads = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/AdsInfo.tsv")
df_ads.registerTempTable("ads")

df_user = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/UserInfo.tsv")
df_user.registerTempTable("user")

df_search = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/SearchInfo.tsv")
df_search.registerTempTable("search")


df_test = sqlContext.read.format("com.databricks.spark.csv").options(delimiter="\t").options(header="true").load("/Volumes/work/data/kaggle/avitocontextadd/testSearchStream.tsv")

#Filter only contextual data from training dataset
trainingdata = df_test.filter(df_test["ObjectType"] == 3)
trainingdata.registerTempTable("train")

#basic sql join on each dataframes in apache spark
#Join AdsInfo with filter trainingdata
df_trainads = sqlContext.sql("SELECT t.*,c.Params,c.Price,c.Title FROM train t JOIN ads c ON t.AdID = c.AdID")
df_trainads.registerTempTable("trainads")


+--------+--------+--------+----------+--------+-------+----------+--------------------+-----+--------------------+
|SearchID|    AdID|Position|ObjectType| HistCTR|IsClick|LocationID|              Params|Price|               Title|
+--------+--------+--------+----------+--------+-------+----------+--------------------+-----+--------------------+
|      25|29614929|       1|         3|0.006885|      0|          |{127:'Детские кол...| 9829|Прогулочная коляс...|
|     138|  123066|       1|         3|0.004306|      0|          |{178:'Для девочек...|  951|               Туфли|
|     273|24856588|       1|         3|0.007524|      0|          |                    |  999|      Берцы Гренадёр|
|     383|24856588|       1|         3|0.015734|      0|          |                    |  999|      Берцы Гренадёр|
|     417|28070889|       1|         3|0.009379|      0|          |{127:'Детские кол...| 9090|Универсальная кол...|
|     638|35572150|       7|         3|0.014952|      0|          |    {5:'Аксессуары'}|  730|     Датчик парковки|
|     679| 6572002|       7|         3|0.036122|      0|          |{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|
|     790|13558903|       1|         3|0.042917|      0|          |{127:'Автомобильн...| 1550|  Детское автокресло|
|     864| 6572002|       7|         3|0.005110|      0|          |{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|
|     940|11089186|       1|         3|0.017910|      0|          |{45:'Столы и стул...|13190|Стол NNDT -T- 426...|
|     956|28070889|       1|         3|0.011285|      0|          |{127:'Детские кол...| 9090|Универсальная кол...|
|    1085|27043682|       1|         3|0.009709|      0|          |{162:'Для студии ...|31699|Пульт Pioneer RMX...|
|    1102| 6572002|       7|         3|0.005411|      0|          |{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|
|    1320|28070889|       1|         3|0.008470|      0|          |{127:'Детские кол...| 9090|Универсальная кол...|
|    1462|28070889|       1|         3|0.007068|      0|          |{127:'Детские кол...| 9090|Универсальная кол...|
|    1541|28070889|       1|         3|0.011422|      0|          |{127:'Детские кол...| 9090|Универсальная кол...|
|    1542|13623847|       7|         3|0.000010|      0|          |{5:'Запчасти', 59...|  150|Решетка радиатора...|
|    1611|35500276|       1|         3|0.034652|      0|          |{5:'Запчасти', 59...| 5941|Муфта сцепления в...|
|    1652| 6016810|       1|         3|0.006780|      0|          |{127:'Велосипеды ...| 7390|Трактор Carmella ...|
|    1782|13558903|       7|         3|0.010950|      0|          |{127:'Автомобильн...| 1550|  Детское автокресло|
+--------+--------+--------+----------+--------+-------+----------+--------------------+-----+--------------------+

#Join further with Location 
df_trainloc = sqlContext.sql("SELECT t.*,c.LocationID FROM trainads t JOIN search c ON t.SearchID = c.SearchID")
df_trainloc.registerTempTable("trainloc")

#data looks like following
+--------+--------+--------+----------+--------+-------+--------------------+-----+--------------------+----------+-----+--------+
|SearchID|    AdID|Position|ObjectType| HistCTR|IsClick|              Params|Price|               Title|LocationID|Level|RegionID|
+--------+--------+--------+----------+--------+-------+--------------------+-----+--------------------+----------+-----+--------+
| 4040558| 8097611|       1|         3|0.002377|      0|{5:'Шины, диски и...| 1590|Штампованный диск...|      3537|    3|      81|
| 4078368|28070889|       1|         3|0.008130|      0|{127:'Детские кол...| 9090|Универсальная кол...|      3537|    3|      81|
| 6616184|13623847|       1|         3|0.006024|      0|{5:'Запчасти', 59...|  150|Решетка радиатора...|      3537|    3|      81|
|17587697|13558903|       7|         3|0.009574|      0|{127:'Автомобильн...| 1550|  Детское автокресло|      1917|    3|      32|
|25188309|28070889|       1|         3|0.017934|      0|{127:'Детские кол...| 9090|Универсальная кол...|      1917|    3|      32|
|28421050|28070889|       1|         3|0.008235|      0|{127:'Детские кол...| 9090|Универсальная кол...|      3537|    3|      81|
|29019362|28070889|       1|         3|0.008969|      0|{127:'Детские кол...| 9090|Универсальная кол...|      1917|    3|      32|
|30211726|29614929|       1|         3|0.009865|      0|{127:'Детские кол...| 9829|Прогулочная коляс...|      3537|    3|      81|
|35275114|28070889|       1|         3|0.009505|      0|{127:'Детские кол...| 9090|Универсальная кол...|      1917|    3|      32|
|36295977|28070889|       1|         3|0.019737|      0|{127:'Детские кол...| 9090|Универсальная кол...|      1917|    3|      32|
|37653743|13558903|       1|         3|0.044174|      0|{127:'Автомобильн...| 1550|  Детское автокресло|      3537|    3|      81|
|39153197| 6572002|       7|         3|0.005116|      0|{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|      3537|    3|      81|
|48231101|28070889|       1|         3|0.009920|      0|{127:'Детские кол...| 9090|Универсальная кол...|      1917|    3|      32|
|50594498|24856588|       1|         3|0.024251|      0|                    |  999|      Берцы Гренадёр|      1917|    3|      32|
|55009263| 6572002|       1|         3|0.024249|      0|{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|      3537|    3|      81|
|60881382| 6572002|       7|         3|0.005411|      0|{5:'Запчасти', 59...|75000|Двигатель 2.7T Au...|      3537|    3|      81|
|64681007|28070889|       1|         3|0.008549|      0|{127:'Детские кол...| 9090|Универсальная кол...|      3537|    3|      81|
|64914576| 8097611|       1|         3|0.004571|      0|{5:'Шины, диски и...| 1590|Штампованный диск...|      3537|    3|      81|
|69839364| 8097611|       1|         3|0.002578|      0|{5:'Шины, диски и...| 1590|Штампованный диск...|      3537|    3|      81|
|74263709|28070889|       1|         3|0.009253|      0|{127:'Детские кол...| 9090|Универсальная кол...|      3537|    3|      81|
+--------+--------+--------+----------+--------+-------+--------------------+-----+--------------------+----------+-----+--------+


