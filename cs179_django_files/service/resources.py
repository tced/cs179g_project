from import_export import resources, fields 
from .models import Person, Users, Business, Reviews 


class PersonResource(resources.ModelResource):
    class Meta:
        model = Person

class UsersResource(resources.ModelResource):
	#userID = fields.Field(column_name = 'userID', attribute="userID")
	#username = fields.Field(column_name = 'username', attribute="username")

	class Meta:
		model = Users
		skip_unchanged = True
		#report_skipped = True 
		exclude = ('id',)
		#import_id_fields = ('userID',)
		fields = ('id', 'userID', 'username') 

class ReviewsResource(resources.ModelResource):
	class Meta:
		model = Reviews

class BusinessResource(resources.ModelResource):
	class Meta:
		model = Business 
		field = {'business_id', 'name', 'avg_stars',
				  'categories'}
		import_id_fields = ['business_id']
		exclude = ('id')
		skip_unchanged = True 
