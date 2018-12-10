from django.shortcuts import get_object_or_404, render, HttpResponse
from tablib import Dataset 
from .models import Business, Reviews, Users
from itertools import chain 
from django.db.models import Avg

#render(request, template file, context)

def service_list(request):
	"""
	Renders the service_list template that list all the currently available polls 
	"""
	business = Business.objects.all()
	review = Reviews.objects.all() 
	user = Users.objects.all() 

	business_list = ''
	review_list = '' 
	review_list1 = ''
	business_results = ''
	user_list = '' 
	
	#grabbing data from the search bar 
	if 'search' in request.GET:
		search_term = request.GET['search']
		business_list = business.filter(name__icontains=search_term)
		review_list = review.filter(bid__in=business_list)
		user_list = user.filter(userID__in=review_list) 


		#prediction_total = review_list.aggregate(Avg('prediction')) 

	context = {'business': business, 'business_list': business_list, 'review_list': review_list }
	return render(request, 'service/service_list.html', context)


def simple_upload(request):
	if request.method == 'POST':
		person_resource = PersonResource()
		dataset = Dataset()
		new_persons = request.FILES['myfile']

		imported_data = dataset.load(new_persons.read())
		result = person_resource.import_data(dataset, dry_run=True)  # Test the data import

	if not result.has_errors():
		person_resource.import_data(dataset, dry_run=False)  # Actually import now

	return render(request, 'service/import.html')