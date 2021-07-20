from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.ann, name="digit"),
    path('cnn/', views.cnn, name="cnn"),
    path('image/', views.image, name="image"),
]

