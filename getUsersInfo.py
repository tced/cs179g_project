from __future__ import print_function

import sys

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col

#****************************************************************
if __name__ == "__main__":
	
	if len(sys.argv) != 2:
		print("Usage: usersInfo <users_data>", file = sys.stderr)
		sys.exit(1)

	#create SparkSession
	spark = SparkSession.builder\
		.appName("Get Users Information ").getOrCreate()

	# Create sql Context so that we can query data files in sql like syntax
   	sqlContext = SQLContext(spark)
	
	spark.sparkContext.setLogLevel("ERROR")
	
	# Read in data and pre-process (from Yelp)
	df = sqlContext.read.json(sys.argv[1])
	print ("Loaded " + str(df.count()) + " records from file with the following schema: ") 
	print(df.schema.names)  
	# Schema: 'average_stars', 'compliment_cool', 'compliment_cute', 'compliment_funny', 'compliment_hot', 
	#	  'compliment_list', 'compliment_more', 'compliment_note', 'compliment_photos', 'compliment_plain', 
	#	  'compliment_profile', 'compliment_writer', 'cool', 'elite', 'fans', 'friends', 'funny', 'name', 
	#	  'review_count', 'useful', 'user_id', 'yelping_since'
	#df.show(2)
	
	# Select the fields we need
	df = df.select(col("user_id").alias("userID"), col("name").alias("username"))
	df.show(2)
	# New Schema: 'userID','username'

	# Write dataframe with results to csv file
	#print("Writing results to file... ")
	#df.coalesce(1).write.format('json').save('/project/outputUsers')
	#df.coalesce(1).write.csv('/project/usersCSV')
	#print("File was written")

	spark.stop()
