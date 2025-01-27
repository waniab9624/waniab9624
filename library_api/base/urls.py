from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search, name='search'),
    path('add/', views.addBook, name='add'),
    path('update/<str:pk>/', views.updateBook, name='update'),
]