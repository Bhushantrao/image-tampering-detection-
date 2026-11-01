from django.urls import path
from . import views

urlpatterns = [

    path('', views.home, name='home'),

    path('upload/', views.home, name='upload'),

    path('history/', views.history, name='history'),

    path('result/<int:pk>/', views.result_detail, name='result_detail'),

    path('analytics/', views.analytics, name='analytics'),

    path('api/analytics-data/', views.analytics_data, name='analytics_data'),
]