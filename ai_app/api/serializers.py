from rest_framework import serializers
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
    # Support either a single image or multiple images
    image = serializers.ImageField(
        required=False,
        help_text="Single image to analyze.")
    images = serializers.ListField(
        child=serializers.ImageField(),
        required=False,
        allow_empty=False,
        help_text="Multiple images to analyze (repeat field in multipart).",
    )
    object_type = serializers.CharField(
        help_text="Which object type to count in the image.",
    )

    def validate(self, attrs):
        single = attrs.get("image")
        many = attrs.get("images")
        if not single and not many:
            raise serializers.ValidationError(
                "Provide either 'image' or 'images'.")
        return attrs


class CorrectionRequestSerializer(serializers.Serializer):
    """Schema for the correction endpoint request body."""
    result_id = serializers.IntegerField(
        help_text="ID of the Result to update.")
    corrected_count = serializers.IntegerField(
        min_value=0, help_text="Corrected number of objects.")


class GenerationRequestSerializer(serializers.Serializer):
    num_images = serializers.IntegerField(min_value=1, default=1)
    max_objects_per_image = serializers.IntegerField(min_value=1, default=1)
    object_types = serializers.ListField(
        child=serializers.CharField(),
        min_length=1
    )
    backgrounds = serializers.ListField(
        child=serializers.CharField(),
        min_length=1
    )
    blur = serializers.FloatField(min_value=0, default=0)
    rotate = serializers.ListField(
        child=serializers.IntegerField(),
        min_length=1
    )
    noise = serializers.FloatField(min_value=0, default=0)