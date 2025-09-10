from django.urls import path
from .views import index, history, set_theme


urlpatterns = [
    path("", index, name="home"),
    path("history/", history, name="history"),
    path("set-theme/", set_theme, name="set_theme"),
]