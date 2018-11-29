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
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: trainNaiveBayes <training data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Train Naive Bayes Model ").getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = spark.read.json(sys.argv[1])
	print (df.schema.names)  # print header names 
	df = df.select("text", "stars") #select the fields we need
	print("Number of reviews: ", df.count())
	df = df.where(col("text").isNotNull()) #Remove reviews that have no text
	df.show(2)

	# Since its not labeled with sentiment, we will assume that a 4 or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, StringType())
	df = df.withColumn("label", my_udf("stars")) # generate a new column called label based on function 
	df.show(5)  # three columns: text, label, and stars
	data = df.select("text", "label") # two column: text and label 
	data.show(5)

	# Remove punctuation and convert to lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView("data") 
	data = spark.table('data').select(lower(col("text")).alias("text"), "label")
	data = data.withColumn("text", punc("text"))
	data = data.withColumn("text", newline("text"))
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
	print ("Number of rows: ", featurizedData.count())
	print("Schema:", featurizedData.schema.names)

	# Obtain the TF-IDF score
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	# Create input data for the model with two columns: label | features
	rescaledData = rescaledData.select("label", "features") 
	print("Row:",  rescaledData.first())
	
	# Separate Sparsevector i.e. instead of {5: 4.297, 30: 1.8509} --> [5,30],[4.296976454524511,1.8508740378963242]
	labeledData = rescaledData.rdd.map(lambda x: LabeledPoint( x["label"], MLLibVectors.fromML(x["features"])))
	print("LabeledPoint Obj:", labeledData.first())

	# Split data into training set and test set
	training, test = labeledData.randomSplit([0.8, 0.2])
	print("training: " ,training.count())
	print("test: ", test.count())

	# Train a naive Bayes model.
	model = NaiveBayes.train(training, 1.0)

	# Save model
	output_dir = '/models/tmp/myNaiveBayesModel'
	model.save(spark, output_dir)
	
	spark.stop()
