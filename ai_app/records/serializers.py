from rest_framework import serializers
from .models import Result

class RecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = "__all__"
