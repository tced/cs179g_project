from __future__ import print_function

import sys

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

#****************************************************************
if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: businessInfo <business_Data> ", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Get Business Information ").getOrCreate()

	# Create sql Context so that we can query data files in sql like syntax
   	sqlContext = SQLContext(spark)
	
	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = sqlContext.read.json(sys.argv[1])
	print ("Loaded " + str(df.count()) + " records from file with the following schema: ")
	print(df.schema.names)
	# Schema: 'address', 'attributes', 'business_id', 'categories', 'city', 'hours', 
	#	  'is_open', 'latitude', 'longitude', 'name', 'neighborhood', 'postal_code', 
	#	  'review_count', 'stars', 'state'
	#df.show(2)
	df = df.select(col("business_id"), col("name"), col("stars").alias("avg_stars"), col("categories"))
	# Final Schema: 'business_id', 'name', 'avg_stars', 'categories' 
	df.show(2)

	# Write dataframe with results to json file
	#print("Writing results to file...")
	#df.coalesce(1).write.csv('/project/businessCVS')
	#print("File was written")

	spark.stop()
