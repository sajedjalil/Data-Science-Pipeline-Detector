import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import scala.util.hashing.{MurmurHash3=>MH3}
import scala.math.{abs, max, min, log}
import collection.mutable.HashMap
import scala.io.{Source}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import java.io.FileWriter
import scala.util.control.Breaks._

// Set higher LogLevel to prevent printing a lot of info
Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("akka").setLevel(Level.WARN)

// Configure Spark driver (when compiling application)
//val conf = new SparkConf()
//    .setMaster("local[*]")
//    .setAppName("AvitoKaggle")
//    .set("spark.executor.memory", "6g")

// Create Spark driver context
//val sc = new SparkContext(conf)

val train_file = "/spark/data/trainSearchStream.tsv"
val test_file = "/spark/data/testSearchStream.tsv"
val submission_input = "/spark/data/sampleSubmission.csv"
val submission_output = "/spark/data/submission.csv"

// Modified to Int convert (will convert empty string to zero)
def toInt(s: String): Int = {
    try {
	s.toInt
    } catch {
	case e: Exception => 0
    }
}

val loss = sc.accumulator(0.0, "LogLoss")
val maxmin = (p:Double) => max(min(p, 1.0 - 1e-14), 1e-14)

//val logloss = (p:Double, y:Double) => - ( y * log(p) + (1-y) * log( 1 - p) )
val logloss = (p:Double, y:Double) => - ( y * log( maxmin(p) ) + (1-y) * log( 1 - maxmin(p) ) )

val train_count = sc.accumulator(1, "Global train examples count")

// Hashing frunction
def hashing (feature: String) : Int = {
    return (abs(MH3.stringHash(feature)) % 20)
}

// Load and parse the training data file

// Train on all clicked objects (label=1) and 3% (~10 millions) of not clicked objects (label=0)
//val allTrainData = sc.textFile(train_file, 15).filter((! _.startsWith("SearchID")))
//val trueTrain = allTrainData.filter(_.endsWith("\t1"))
//val falseTrain = allTrainData.filter(_.endsWith("\t0")).sample(false, 0.03, System.currentTimeMillis().toInt)
//val trainData = falseTrain.union(trueTrain)

// Train on 10% of all data
val trainData = sc.textFile(train_file, 15).filter((! _.startsWith("SearchID"))).sample(false, 0.2, System.currentTimeMillis().toInt)

// Train on all data
//val trainData = sc.textFile(train_file, 15).filter((! _.startsWith("SearchID")))

val parsedTrainData = trainData.map { line => 
    val part = line.split("\t", 6)
//    LabeledPoint(toInt(part(5)), Vectors.dense(hashing("AdID_" + part(1)), hashing("Position_" + part(2)), hashing("ObjectType_" + part(3))))
    LabeledPoint(toInt(part(5)), Vectors.dense(part(1).toDouble, part(2).toDouble, part(3).toDouble))
}.cache()

//Load saved model
//val trained = LogisticRegressionModel.load(sc, "AvitoKaggleModel")

// Build classification model 
//val model = new LogisticRegressionWithSGD()
val model = new LogisticRegressionWithLBFGS().setNumClasses(2)
//model.optimizer.setRegParam(0.001).setNumIterations(1)
//model.optimizer.setNumIterations(100)

// Run model on training set
val trained = model.run(parsedTrainData)

// Switch to output probabilities instead of 0/1 values
trained.clearThreshold()

// Save model
//trained.save(sc, "AvitoKaggleModel")

// Check log loss on training dataset
val predictedTrainData = parsedTrainData.map { case LabeledPoint(label, features) => 
    val prediction = trained.predict(features)
    loss += logloss(prediction, label)
    train_count += 1
    (label, prediction)
}

// Save predictions of training dataset with LogLoss (for algorithm evaluation)
predictedTrainData.saveAsTextFile("TrainDatasetPredictions")

val total_loss = loss.value/train_count.value

parsedTrainData.unpersist()

// Load and parse the test/real data file, then predict
val testData = sc.textFile(test_file, 15).filter((! _.startsWith("ID")))
val predictedTestData = testData.map { line => 
    val part = line.split("\t")
    val id = part(0)
//    val features = Vectors.dense(hashing("AdID_" + part(2)), hashing("Position_" + part(3)), hashing("ObjectType_" + part(4)))
    val features = Vectors.dense(part(2).toDouble, part(3).toDouble, part(4).toDouble)
    val prediction = trained.predict(features)
    (id, prediction)
}

// Save predicted test dataset
//predictedTestData.saveAsTextFile("TestDatasetPredictions")

// Collect all predicted labels to master machine
val localPredictedTestData = predictedTestData.collectAsMap()

predictedTestData.unpersist()

// Save predicted results as a submission file (include only needed elements)
val sampleFile = Source.fromFile(submission_input)
val submissionOutput = new FileWriter(submission_output, true)
submissionOutput.write("ID,IsClick\n")

for (sampleLine <- sampleFile.getLines().filter((! _.startsWith("ID")))) {
    val columns = sampleLine.split(",").map(_.trim)
    val id = columns(0).toString
    submissionOutput.write(id + "," + localPredictedTestData(id).toString + "\n")
}

submissionOutput.close()
