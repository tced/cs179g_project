cs179g_project Fall 2018
Tiffany Cedeno
Andrea Cruz Castillo
Patrick Nguyen

In order to run the Django files, you must be in the directory: cs179_django_files
  use the command: python3 manage.py runserver 
  The terminal will say that the code is running. 
  On an browser (i.e. Firefox or google chrome), put in the url: http://127.0.0.1:8000
      Since the website is created locally, it will run via the local port. 

To run the files in mlCode execute the following command:

	spark-submit --driver-memory 5g <file in mlCode> <path to input file in hdfs>
	
	For example:
		spark-submit --driver-memory 5g trainNaiveBayes.py /yelp_academic_dataset_review.json

	The following files are the input files for the respetive pyspark scripts
	
		/yelp_academic_dataset_review.json	trainNaiveBayes.py
							testNaiveBayes.py
							trainLogisticRegression.py
							testLogisticRegression.py
		
		/yelp_academic_dataset_user.json	getUsersInfo.py
		
		/yelp_academis_dataset_business.json	getBusinessInfor.py	

	Note: you must add --driver-memory 5g to the command, else the program will crash due
		to limited memory access.

The Yelp dataset can be found at https://www.yelp.com/dataset

All of the output files from the mlCode scripts will be saved to hdfs. You can check they
were created by typing
	
	hdfs dfs -ls /<path>
      
    

