from django.conf import settings 
from django.conf.urls import url 
from django.conf.urls.static import static 
from . import views 

#how to setup the index for each individual app section 
urlpatterns = [
	url(r'^$', views.index, name='index'), 
]

#only to check settings, DEBUG 
if settings.DEBUG:
	urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) 