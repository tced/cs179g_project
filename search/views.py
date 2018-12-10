from django.shortcuts import render
from django.template import loader 
from django.http import HttpResponse

# Create your views here.
#takes the request, and sends back an HTTP response 
def index(request): 
	template = loader.get_template('search/homepage.html')
	return HttpResponse(template.render())  