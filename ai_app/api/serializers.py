from rest_framework import serializers
from records.models import Result

class ResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = Result
        fields = [
            "id", "image", "object_type",
            "predicted_count", "corrected_count",
            "meta", "status", "created_at", "updated_at"
        ]
        read_only_fields = ["id", "predicted_count", "meta", "status", "created_at", "updated_at"]

    def validate_image(self, value):
        # quick validation: must be an image
        if not value.content_type.startswith("image/"):
            raise serializers.ValidationError("Uploaded file must be an image.")
        return value
