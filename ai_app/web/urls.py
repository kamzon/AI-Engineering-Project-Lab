from django.urls import path
from .views import index, history


urlpatterns = [
    path("", index, name="home"),
    path("history/", history, name="history"),
]