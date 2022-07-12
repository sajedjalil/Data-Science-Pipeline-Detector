import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# A simple Apache Spark MLLib GBTClassifier with CrossValidation example

spark = SparkSession.builder.appName("Spark for PSSDP").getOrCreate()

spark.sparkContext.setCheckpointDir('checkpoint/')

train_data = spark.read.csv("../input/train.csv", header="true", inferSchema="true")
train_data = train_data.na.fill(-1)
test_data = spark.read.csv("../input/test.csv", header="true", inferSchema="true")
test_data = test_data.na.fill(-1)

print("loaded training (%d, %d) and testing (%d, %d)"%(train_data.count(), len(train_data.columns), test_data.count(), len(test_data.columns)))

# drop ps_calc_* columns
ps_calc_columns = [col for col in train_data.columns if col.startswith("ps_calc_")]
train_data = train_data.drop(*ps_calc_columns)
test_data = test_data.drop(*ps_calc_columns)

print("after preprocessing training (%d, %d) and testing (%d, %d)"%(train_data.count(), len(train_data.columns), test_data.count(), len(test_data.columns)))


ignore = ['id', 'target']
assembler = VectorAssembler(
    inputCols=[x for x in train_data.columns if x not in ignore],
    outputCol='features')

train_data = (assembler.transform(train_data).select("target", "features"))

# with 250 iterations, GINI is around ~0.276 for submission
iteration = 250
gbt = GBTClassifier(labelCol="target",
                    featuresCol="features", maxIter=iteration)

evaluator = BinaryClassificationEvaluator(labelCol="target")

# no parameter search
paramGrid = ParamGridBuilder().build()

# 6-fold cross validation
crossval = CrossValidator(
    estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=6)


model = crossval.fit(train_data)

print("trained GBT classifier:%s" % model)

# display CV score
auc_roc = model.avgMetrics[0]
print("AUC ROC = %g" % auc_roc)
gini = (2 * auc_roc - 1)
print("GINI ~=%g" % gini)


# prepare submission
predictions = model.transform(
    assembler.transform(test_data).select("features"))

print("predicted testing data")

# extract ids from test data
ids = test_data.select("id").rdd.map(lambda x: int(x[0]))

# we should provide probability of 2nd class
targets = predictions.select("probability").rdd.map(lambda x: float(x[0][1]))

# create data frame consists of id and probabilities
submission = spark.createDataFrame(ids.zip(targets), StructType([StructField(
    "id", IntegerType(), True), StructField("target", FloatType(), True)]))

# store results after coalescing
submission.coalesce(1).write.csv('%d-%g-%g.csv' %
                                 (iteration, auc_roc, gini), header="true")


print("exported predictions for submission")

spark.stop()