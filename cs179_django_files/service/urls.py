from django.urls import path
from . import views

app_name ="service"

urlpatterns = [
    #creates a path time using the function from views
    #need to include this in the root directory in package level directory
    path('list/', views.service_list, name="list"),
]
