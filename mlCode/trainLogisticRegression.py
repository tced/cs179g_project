from __future__ import print_function

import sys
import re
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import lower, col, udf
from pyspark.sql.types import *

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer, RegexTokenizer

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
#****************************************************************
#	User Defined Functions

def generateLabel(v):
	if v >= 4:
		return 1
	else:
		return 0
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: trainLogisticRegression <training data>", file = sys.stderr)
		sys.exit(1)

	# Create SparkSession
	spark = SparkSession.builder\
		.appName("Train Logistic Regression Model ").getOrCreate()

	# Create sql Context so that we can query data files in sql like syntax
	sqlContext = SQLContext(spark)

	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in training data (from Yelp)  and pre-process it
	df = sqlContext.read.json(sys.argv[1])
	print ("Loaded " + str(df.count()) + " records with the following schema " + str(df.schema.names))  # print header names 
	df = df.select("business_id", "text", "stars") #select the fields we need
	df = df.where(col("text").isNotNull()) #Remove reviews that have no text
	df.show(2)

	# Since its not labeled with sentiment, we will assume that a 4 or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, IntegerType())
	data = df.withColumn("label", my_udf("stars")) # generate a new column called label based on function 
	data.show(2)

	# Remove punctuation and convert to lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView("data") 
	data = spark.table('data').select("business_id", lower(col("text")).alias("text"), "label", "stars")
	data = data.withColumn("text", punc("text"))
	data = data.withColumn("text", newline("text"))
	data.show(2)

	# Generate tokens 
	tokenizer = Tokenizer(inputCol="text", outputCol="words")
	wordsData = tokenizer.transform(data)

	# Create Ngrams
	ngram = NGram(n = 2, inputCol= "words", outputCol = "bigrams")
	wordsData = ngram.transform(wordsData)

	# inputCol can be "words" or "bigrams"
	hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=3600)
	featurizedData = hashingTF.transform(wordsData)
	#print ("Number of rows: ", featurizedData.count())
	print("Schema:", featurizedData.schema.names)

	# Obtain the TF-IDF score
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	#print("Row:",  rescaledData.first())
	
	# Split data into training set and test set
	training, test = rescaledData.randomSplit([0.8, 0.2])
	#print("training: " ,training.count())
	#print("test: ", test.count())

	# Train a Logistic Regression model.
	lr = LogisticRegression(featuresCol = "features", labelCol = "label", predictionCol = "prediction")
	model = lr.fit(training)

	predictions = model.transform(test)
	predictions = predictions.drop("predictions", "rawFeatures", "features", "rawPrediction", "probability")	
	predictions.show(10)
	
	#print("Calculating mean squared error...")
	#evaluator = RegressionEvaluator(labelCol = "label", predictionCol = "prediction", metricName = "mse")
	#mse = evaluator.evaluate(predictions)
	#print("Mean Squared Error = " + str(mse))
	
	print("Calculating accuracy...")
   	evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName = "accuracy")
   	accuracy = evaluator.evaluate(predictions)
   	print("Test set accuracy = " + str(accuracy))
	# Save model
	print("Saving model...")
	output_dir = '/models/tmp/myLogisticRegressionModel'
	model.save(output_dir)
	print("Model saved")
	
	spark.stop()
