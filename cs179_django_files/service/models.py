from django.db import models

# Create your models here.
class Person(models.Model):
    name = models.CharField(max_length=30)
    email = models.EmailField(blank=True)
    birthdate = models.DateField()
    location = models.CharField(max_length=100, blank=True)

    def __str__(self):
    	return self.name + " " + self.email 

class Users(models.Model): 
	userID = models.CharField(max_length = 30)
	username = models.CharField(max_length = 250)

	#override 
	#class Meta: 
	def __str__(self):
		return self.username

class Business(models.Model): 
	bid = models.CharField(max_length = 100) 
	name = models.CharField(max_length = 50) 
	avgstars = models.CharField(max_length = 3)
	categories = models.CharField(max_length = 200)

class Reviews(models.Model): 
	bid = models.ForeignKey(Business, on_delete=models.CASCADE) 
	text = models.CharField(max_length = 100) 
	stars = models.CharField(max_length = 3)
	prediction = models.CharField(max_length = 3, default='0') 
	userID = models.ForeignKey(Users, on_delete=models.CASCADE)
	rid = models.CharField(max_length = 30) 


	#def prediction():
	#	return aggregate(Avg(prediction)) 

	#def __str__(self):
	#	return self.text
	