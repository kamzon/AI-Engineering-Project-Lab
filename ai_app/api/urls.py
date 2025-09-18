from django.urls import path
from .views import CountView, CorrectionView, GenerateView, GeneratePreviewView, GenerateFinetuneView

urlpatterns = [
    path("count/", CountView.as_view(), name="count"),
    path("correct/", CorrectionView.as_view(), name="correct"),
    path("generate/", GenerateView.as_view(), name="generate"),
    path("generate/preview/", GeneratePreviewView.as_view(), name="generate-preview"),
    path("generate/finetune/", GenerateFinetuneView.as_view(), name="generate-finetune"),
]