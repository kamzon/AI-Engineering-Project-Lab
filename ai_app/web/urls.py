from django.urls import path
from .views import index, history, set_theme, object_types


urlpatterns = [
    path("", index, name="home"),
    path("history/", history, name="history"),
    path("set-theme/", set_theme, name="set_theme"),
    path("object-types/", object_types, name="object_types"),
]