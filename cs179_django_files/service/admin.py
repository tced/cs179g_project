from __future__ import unicode_literals 

from import_export.admin import ImportExportModelAdmin
from import_export import resources 
from django.contrib import admin
# Register your models here.
from .models import Person, Users, Business, Reviews

from .resources import UsersResource
#grabbing database, Users, so we have access to it
@admin.register(Person)
@admin.register(Users)
@admin.register(Reviews) 
@admin.register(Business)

#admin.register(Person)
class PersonAdmin(ImportExportModelAdmin):
    pass

class ReviewsAdmin(ImportExportModelAdmin):
	pass

class UsersAdmin(ImportExportModelAdmin):
	pass

#class UsersPostAdmin(admin.ModelAdmin):
#	class Meta:
#		list_display = ["username", "userID"]

class BusinessAdmin(ImportExportModelAdmin):
	#resource_class = BusinessResource
	#list_display = {'id', 'business_id', 'name'}
	pass 

#admin.site.register(Users, UsersPostAdmin)
