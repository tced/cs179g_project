from __future__ import print_function
import csv
import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import *
import string
import re
import nltk
from nltk.corpus import stopwords



def removePunctuation(myString):
	myString = re.sub(r'[^\w\s\d-]', '', myString)
	return myString

def getTokens(row):
	words = row.lower().split()
	return [ a + " " + b for a, b in zip(words, words[1:])]

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: getReviews <file>", file = sys.stderr)
		sys.exit(1)

	spark = SparkSession\
		.builder\
		.appName("PythonGetReviews")\
		.getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")

	sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)

#*************************************************************************************************************
	#TO TEST GETTOKENS AND REMOVE PUNCTUATION COMMENT OUT THE NEXT TWO LINES AND UNCOMMENT THE FIRST LINE AFTER ****
	yelp_df = sqlContext.read.json(sys.argv[1]) 
	yelp_df.show(1, False)
	
	training_df = spark.read.load(sys.argv[2], format = "csv", sep = "\t", header = "true")
	header = next(training_df)
	data_df.show(1, False)
#*************************************************************************************************************

#       yelp_df = sqlContext.read.json(sys.argv[1]).select("text").rdd.map(lambda r: r[0])
#	----------REMOVE STOP WORDS HERE-----------------------

#	with open(sys.argv[3]) as f:
#	   stopwords = f.read().splitlines()
#

#	lines = yelp_df.flatMap(getTokens)\
#		.filter(lambda x: x not in stopwords)\
#		.map(lambda x: removePunctuation(x))\
#		.map(lambda x:(x,1))

#	counts = lines.reduceByKey(add)
#	wordCount = counts.collect()

#	for (word,count) in wordCount:
#		print("%s:%i"%(word,count))

	spark.stop()
