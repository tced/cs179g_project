from __future__ import print_function
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
import string
import re
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
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
#****************************************************************


if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: buildModel <training data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Build Sentiment Analysis Model ").getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")
	
	# 1 - Read in the training data and pre-process (from Yelp)
	data = spark.read.csv(sys.argv[1], sep = "\t", inferSchema = "true", header = "true")
	print (data.schema.names)  # print header names 
	#not present *****  reviewDF = df.select('text', 'stars')
	print("Number of reviews: ", data.count())
	# may not need **** df = df.where(col("Review").isNotNull()) #Remove reviews that have no text
	data.show(2)

	'''# 2 - Since its not labeled with sentiment, we will assume that a 4 or above is positive
	my_udf = F.UserDefinedFunction(generateLabel, T.StringType())
	df = df.withColumn('label', my_udf('stars')) # generate a new column called label based on function 
	df.show(5)  # three columns: text, label, and stars
	data = df.select("text", "label") # two column: text and label 
	data.show(5)'''

	# 3 - remove punctuation and lowercase
	punc = udf(lambda x: re.sub(r'[^\w\s\d]', '', x))
	newline = udf(lambda y: re.sub('[\n\t]',' ',y))
	data.createOrReplaceTempView('data') 
	data = spark.table('data').select(lower(col('Review')).alias('text'), 'Liked')
	data = data.withColumn('text', punc('text'))
	data = data.withColumn('text', newline('text'))
	data.show(5)

	# 4 - generate tokens 
	tokenizer = Tokenizer(inputCol="text", outputCol="words")
	wordsData = tokenizer.transform(data)

	#create Ngrams
	#ngram = NGram(n = 2, inputCol= "words", outputCol = "ngrams")
	#wordsData = ngram.transform(wordsData)

	#changed inputCol = "words"
	hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10)
	featurizedData = hashingTF.transform(wordsData)
	print ("Number of rows: ", featurizedData.count())
	print("Schema:", featurizedData.schema.names)
	# alternatively, CountVectorizer can also be used to get term frequency vectors

	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	rescaledData = rescaledData.select("Liked", "features")  # data 
	print("Row:",  rescaledData.first())
	
	#separate Sparsevector i.e. instead of {5: 4.297, 30: 1.8509} --> [5,30],[4.296976454524511,1.8508740378963242]
	labeledData = rescaledData.rdd.map(lambda x: LabeledPoint(x["Liked"], MLLibVectors.fromML(x["features"])))

	print("LabeledPoint Obj:", labeledData.first())

	training, test = labeledData.randomSplit([0.8, 0.2])
	
	# Train a naive Bayes model.
	model = NaiveBayes.train(training, 1.0)

	# Make prediction and test accuracy.
	predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
	accuracy = 1.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / test.count()
	print('model accuracy {}'.format(accuracy))


	# Save and load model
	#output_dir = 'target/tmp/myNaiveBayesModel'
	#shutil.rmtree(output_dir, ignore_errors=True)
	#model.save(sc, output_dir)'''

	spark.stop()
