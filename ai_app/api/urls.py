from django.urls import path
from .views import CountView, CorrectionView

urlpatterns = [
    path("count/", CountView.as_view(), name="count"),
    path("correct/", CorrectionView.as_view(), name="correct"),
    path("generate/", CorrectionView.as_view(), name="generate"),  
]