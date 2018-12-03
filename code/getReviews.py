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

#****************************************************************
def removePunctuation(myString):
	myString = re.sub(r'[^\w\s\d]', '', myString)
	return myString

def createBigrams(row):
	words = row.lower().split()
	return [ a + " " + b for a, b in zip(words, words[1:])]
#****************************************************************


if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("Usage: getReviews <file>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("PythonGetReviews").getOrCreate()

	spark.sparkContext.setLogLevel("ERROR")
	
	#read stopwords File
	with open(sys.argv[3]) as f:
		stopWords = f.read().splitlines()

	#read and process yelp_dataset 
	yelp = spark.read.json(sys.argv[1]).select("text")
	yelp.show(1, False)
	yelpLines = yelp.rdd.map( lambda r: r[0])\
			.flatMap(createBigrams)\
			.filter(lambda x: x not in stopWords)\
			.map(lambda x: removePunctuation(x))\
			.map(lambda x: (x,1))

	#read and process training set from kaggle
	kaggle = spark.read.csv(sys.argv[2], sep = "\t", inferSchema = "true", header = "true")
	kaggleLines = kaggle.rdd.map(lambda r: r[0])\
			.flatMap(createBigrams)\
			.filter(lambda x: x not in stopWords)\
			.map(lambda x: removePunctuation(x))\
                        .map(lambda x: (x,1))

#	yelpCounts = yelpLines.reduceByKey(add)\
#                                .map(lambda (a,b):(b,a))\
#                                .sortByKey(0,1)\
#                                .map(lambda (a,b):(b,a))

#	yelpWordCount = yelpCounts.collect()
	
#	for (word, count) in yelpWordCount:
#		print("Yelp: %s	:%i"%(word, count))


	kaggleCounts= kaggleLines.reduceByKey(add)\
				.map(lambda (a,b):(b,a))\
				.sortByKey(0,1)\
				.map(lambda (a,b):(b,a))

	wordCount = kaggleCounts.collect()

	for (word,count) in wordCount:
		print("%s:%i"%(word,count))

	spark.stop()
