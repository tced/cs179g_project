from __future__ import print_function

import sys
import re
import pyspark.sql.functions as F

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import lower, col, lit, udf
from pyspark.sql.types import *

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer, RegexTokenizer

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#****************************************************************
# Imcomplete implementation. Some reviews get labeled as more 
# than one category thus providing a very low accuracy. Must
# assign the correct label to each review. i.e. majority vote

#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: coolFunnyUseful <training data>", file = sys.stderr)
		sys.exit(1)

	# Create SparkSession
	spark = SparkSession.builder\
		.appName("Classify Cool Funny or Useful").getOrCreate()

	# Create sql Context so that we can query data files in sql like syntax
	sqlContext = SQLContext(spark)

	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in training data (from Yelp)  and pre-process it
	df = sqlContext.read.json(sys.argv[1])
	print ("Loaded " + str(df.count()) + " records with the following schema " + str(df.schema.names))  # print header names 

	# Drop columns we don't need (date, stars, user_id, review_id)
	df = df.drop("date", "stars", "user_id", "review_id")

	# Select reviews voted as funny
	funnyRevs = df.where(col("funny") != 0)
	print("Number of funny reviews = ", funnyRevs.count())
	# Take a sample of funny reviews to avoid imbalance
	funnySample = funnyRevs.sample( withReplacement = False, fraction = 0.79)
	print("funnySample: ", funnySample.count())
	# Add a label column for classification 0 => funny, 1 => cool, 2 => useful
	funnySample = funnySample.withColumn("label", lit(0))
	funnySample.show(2)

	# Select reviews voted as cool
	coolRevs = df.where(col("cool") != 0)
	print("Number of cool reviews = ", coolRevs.count())
	# Take a sample of cool reviews to avoid imbalance
	coolSample = coolRevs.sample( withReplacement = False, fraction = 0.65)
	print("coolSample: ", coolSample.count())
	# Add a label column for classification 0 => funny, 1 => cool, 2 => useful
	coolSample = coolSample.withColumn("label", lit(1))
	coolSample.show(2)	
	
	# Select reviews voted as useful
	usefulRevs = df.where(col("useful") != 0)
	print("Number of useful reviews = ", usefulRevs.count())
	# Take a sample of useful reviews to avoid imbalance
	usefulSample = usefulRevs.sample( withReplacement = False, fraction = 0.35)
        print("usefulSample: ", usefulSample.count())
	# Add a label column for classification 0 => funny, 1 => cool, 2 => useful
	usefulSample = usefulSample.withColumn("label", lit(2))
	usefulSample.show(2)
	
	# Union all three samples to create one dataframe
	data = funnySample.union(coolSample).union(usefulSample)
	data.show(5)

	# Remove punctuation and convert to lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView("data") 
	data = spark.table('data').select("business_id","funny", "cool", "useful", lower(col("text")).alias("text"), "label")
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

	# Train a naive Bayes model.
	print("Begin training...")
	nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial")
	model = nb.fit(training)
	print("Training completed.")

	predictions = model.transform(test)
	predictions = predictions.drop("predictions", "rawFeatures", "features", "rawPrediction", "probability")	
	predictions.show(10)
	
	print("Calculating accuracy...")
	evaluator = MulticlassClassificationEvaluator(labelCol = "label", predictionCol = "prediction", metricName = "accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test set accuracy = " + str(accuracy))

	# Save model
	print("Saving model...")
	#output_dir = '/models/tmp/myNaiveBayesModel'
	#model.save(output_dir)
	print("Model saved")
	
	spark.stop()
