from __future__ import print_function
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
import re
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
#from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import pyspark.sql.functions as F
#import pyspark.sql.types as T
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector
from pyspark.ml.linalg import Vector as MLVector, Vectors as MLVectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from pyspark.mllib.regression import  LabeledPoint
import json
from pyspark.sql.functions import lower, col
#****************************************************************
#	User Defined Functions

def generateLabel(v):
	if v >= 3.8:
		return 1
	else:
		return 0
def createCol(l):
	l = l.map(lambda x:x[0])
	return l
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: buildModel <training data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Build Sentiment Analysis Model ").getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = spark.read.json(sys.argv[1])
	print (df.schema.names)  # print header names 
	df = df.select('business_id','text', 'stars') #select the fields we need
	print("Number of reviews: ", df.count())
	df = df.where(col("text").isNotNull()) #Remove reviews that have no text
	df.show(2)

	# Since its not labeled with sentiment, we will assume that a 4 or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, StringType())
	df = df.withColumn('label', my_udf('stars')) # generate a new column called label based on function 
	df.show(5)  # three columns: text, label, and stars
	data = df.select("business_id","text", "label") # two column: text and label 
	data.show(5)

	# Remove punctuation and convert to lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d-]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView('data') 
	data = spark.table('data').select('business_id',lower(col('text')).alias('text'), 'label')
	data = data.withColumn('text', punc('text'))
	data = data.withColumn('text', newline('text'))
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
	
	#save information to file => [businessID, review, label, words] -> will append prediction later
	#featurizedData.coalesce(1).write.json( '/project/output.json') -- does not return
	#https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=read%20csv --> jdbc(url, table, mode=None, properties=None)[source] or csv to append prediction later

	# alternatively, CountVectorizer can also be used to get term frequency vectors

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
	model = NaiveBayes.train(training, 1.0) #addr

	# Make prediction and test accuracy.
	predictionAndLabel = test.map(lambda p: (int(model.predict(p.features)), int(p.label)))
	print("predictionAndLabel: ", predictionAndLabel.count())
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
	print('model accuracy {}'.format(accuracy))

	# Save model
	output_dir = '/models/tmp/myNaiveBayesModel'
	#shutil.rmtree(output_dir, ignore_errors=True)
	model.save(spark, output_dir)

	spark.stop()
