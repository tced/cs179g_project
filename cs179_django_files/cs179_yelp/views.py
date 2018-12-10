from django.shortcuts import render 

def home(request):
	#creates key value pairs
	context = {}
	return render(request, 'home.html', context)
