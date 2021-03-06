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

from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
#****************************************************************
#	User Defined Functions

def generateLabel(v):
	if v >= 3:
		return 1
	else:
		return 0
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: testLogisticRegression <test data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Test Logistic Regression Model ").getOrCreate()

	# Create sql Context so that we can query data files in sql like syntax
   	sqlContext = SQLContext(spark)
	
	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = sqlContext.read.json(sys.argv[1])
	print ("Loaded " + str(df.count()) + " records from file with the following schema: " + str(df.schema.names))  # print header names 
	df = df.select("business_id", "review_id", "user_id", "text", "stars") #select the fields we need
	df = df.where(col("text").isNotNull()) #Remove reviews that have no text
	df.show(2)
	# Since its not labeled with sentiment, we will assume that a value 4 or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, IntegerType())
	print("Creating label for each review")
	data = df.withColumn("label", my_udf("stars")) # generate a new column called label based on function 
	data.show(2)  # three columns: text, label, and stars

	# Remove punctuation and convert to lowercase
	print("Removing punctuation and coverting to lowercase")
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView("data") 
	data = spark.table("data").select("business_id", "review_id", "user_id", lower(col("text")).alias("text"), "label", "stars")
	data = data.withColumn("text", punc("text"))
	data = data.withColumn("text", newline("text"))
	data.show(2)

	# Generate tokens 
	tokenizer = Tokenizer(inputCol="text", outputCol="words")
	wordsData = tokenizer.transform(data)

	# Create Ngrams, specify n
	ngram = NGram(n = 2, inputCol= "words", outputCol = "bigrams")
	wordsData = ngram.transform(wordsData)

	# inputCol can be "words", "bigrams" or whatever the outputCol of Ngram is
	hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=3600)
	featurizedData = hashingTF.transform(wordsData)
	
	# Obtain the TF-IDF score
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
		
	# Load Logistic Regression model.
	print("Loading Logistic Regression Model...")
	model = LogisticRegressionModel.load('/models/tmp/myLogisticRegressionModel')

	# Make prediction and test accuracy.
	print("Logistic Regression loaded. Begin testing...")
	predictions = model.transform(rescaledData)
	print("Testing done.")

	# Compute mean squared error
	#print("Calculating mean squared error...")
   	#evaluator = RegressionEvaluator(labelCol = "label", predictionCol = "prediction", metricName = "mse")
   	#mse = evaluator.evaluate(predictions)
   	#print("Mean Squared Error = " + str(mse))	
	
	# Compute accuracy
	print("Calculating accuracy...")
	evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName = "accuracy")
   	accuracy = evaluator.evaluate(predictions)
   	print("Test set accuracy = " + str(accuracy))

	# Format output data
	resultsDF = predictions.drop("predictions", "rawFeatures", "features", "rawPrediction", "probability", "words", "bigrams", "label")
   	resultsDF.show(2)

	# Write dataframe with results to csv file
	# Schema: business_id| review_id| user_id| text| stars| prediction
	print("Writing results to file...")
	#resultsDF.coalesce(1).write.format('json').save('/project/outputLRjson')
	#resultsDF.coalesce(1).write.csv('/project/outputLR')
	print("File was written")
	
	spark.stop()
