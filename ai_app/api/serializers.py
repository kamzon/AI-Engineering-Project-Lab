from rest_framework import serializers
from django.conf import settings
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


class CountRequestSerializer(serializers.Serializer):
    """Schema for the count endpoint request body."""
    image = serializers.ImageField(
        help_text="Image file to analyze. Accepts common image formats.")
    object_type = serializers.ChoiceField(
        choices=settings.OBJECT_TYPES,
        help_text="Which object type to count in the image.",
    )


class CorrectionRequestSerializer(serializers.Serializer):
    """Schema for the correction endpoint request body."""
    result_id = serializers.IntegerField(
        help_text="ID of the Result to update.")
    corrected_count = serializers.IntegerField(
        min_value=0, help_text="Corrected number of objects.")
