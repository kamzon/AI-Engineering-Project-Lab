from rest_framework import viewsets
from .models import Result
from .serializers import RecordSerializer

class RecordViewSet(viewsets.ModelViewSet):
    queryset = Result.objects.all()
    serializer_class = RecordSerializer
