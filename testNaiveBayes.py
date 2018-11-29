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

from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.mllib.regression import  LabeledPoint
#****************************************************************
#	User Defined Functions

def generateLabel(v):
	if v >= 3.8:
		return 1
	else:
		return 0

def with_column_index(sdf, label):
	new_schema = StructType(sdf.schema.fields + [StructField(label, LongType(), False),])
	return sdf.rdd.zipWithIndex().map(lambda row: row[0] + (row[1],)).toDF(schema=new_schema)
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: testNaiveBayes <test data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Test Naive Bayes Model ").getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = spark.read.json(sys.argv[1])
	print ("Loaded file with the following schema: ",df.schema.names)  # print header names 
	df = with_column_index(df, "index")
	datafrm = df.select("text", "stars") #select the fields we need
	print("Number of reviews: ", datafrm.count())
	datafrm = datafrm.where(col("text").isNotNull()) #Remove reviews that have no text
	print("Selecting fields")

	# Since its not labeled with sentiment, we will assume that a value (3.8) or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, StringType())
	datafrm = datafrm.withColumn("label", my_udf("stars")) # generate a new column called label based on function
	print("Creating label for each review") 
	datafrm.show(5)  # three columns: text, label, and stars
	data = datafrm.select("text", "label") # two column: text and label 
	print("Selecting only the necessary fields")
	data.show(5)

	# Remove punctuation and convert to lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView("data") 
	data = spark.table("data").select(lower(col("text")).alias("text"), "label")
	data = data.withColumn("text", punc("text"))
	data = data.withColumn("text", newline("text"))
	print("Removed punctuation and converted to lowercase")
	data.show(4)

	# Generate tokens 
	tokenizer = Tokenizer(inputCol="text", outputCol="words")
	wordsData = tokenizer.transform(data)

	# Create Ngrams
	#ngram = NGram(n = 2, inputCol= "words", outputCol = "ngrams")
	#wordsData = ngram.transform(wordsData)

	#changed inputCol = "words"
	hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=3600)
	featurizedData = hashingTF.transform(wordsData)
	
	# Obtain the TF-IDF score
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	# Create input data for the model with two columns: label | features
	rescaledData = rescaledData.select("label", "features")
	
	# Separate Sparsevector i.e. instead of {5: 4.297, 30: 1.8509} --> [5,30],[4.296976454524511,1.8508740378963242]
	labeledData = rescaledData.rdd.map(lambda x: LabeledPoint( x["label"], MLLibVectors.fromML(x["features"])))
		
	# Load Naive Bayes model.
	print("Loading Naive Bayes Model...")
	model = NaiveBayesModel.load(spark, '/models/tmp/myNaiveBayesModel')

	# Make prediction and test accuracy.
	print("Naive Bayes Model loaded. Starting testing...")
	predictionAndLabel = labeledData.map(lambda p: (model.predict(p.features), p.label))
	print("Testing completed.")
	#accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / labeledData.count()------
	#print("Model accuracy {}".format(accuracy)) ----

	# Create dataframe with desired content for output
	resultsDF = df.select("business_id", "review_id", "text", "index")
	resultsDF.show(10)

	# Define schema for prediction and label dataframe
	schema = StructType([\
			StructField("prediction", StringType(), False),\
			StructField("label", StringType(), False)])	
	
	# Convert RDD of prediction and labels to dataframe 
	predictionAndLabelDF = spark.createDataFrame(predictionAndLabel, schema)
	# Add index to prediction and label dataframe to be able to join tables by index
	print("Adding indexes to prediction and label")
	predictionWithIndex = with_column_index(predictionAndLabelDF, "number")
	predictionWithIndex.show(9)
	# Join dataframes to obtain results as -> [business_id, review_id, text, index, prediction]
	print("Joining tables")

	# *****************ERROR HAPPENS HERE**************
	resultsDF = resultsDF.join(predictionWithIndex, resultsDF.index == predictionWithIndex.number, 'inner').drop("number").sort(resultsDF.index, asc = True)

	print("Joined tables")
	resultsDF.show(8)
	# Write dataframe with results to json file
	resultsDF.coalesce(1).write.format('json').save('/project/output')
	print("File was written")
	
	spark.stop()
