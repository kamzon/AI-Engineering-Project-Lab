from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import RecordViewSet

router = DefaultRouter()
router.register(r'records', RecordViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
