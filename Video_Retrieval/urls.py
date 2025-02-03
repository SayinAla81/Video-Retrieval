# myproject/urls.py
from django.contrib import admin
from django.urls import path
from Video_Retrieval.views import ImageSearchView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('search/', ImageSearchView.as_view(), name='image_search'),
]