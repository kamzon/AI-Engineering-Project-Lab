from rest_framework import serializers
from rest_framework import serializers
from django.conf import settings
from records.models import Result
from PIL import Image, UnidentifiedImageError

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
    image = serializers.FileField(
        required=False,
        help_text="Single image to analyze.")
    images = serializers.ListField(
        child=serializers.FileField(),
        required=False,
        allow_empty=False,
        help_text="Multiple images to analyze (repeat field in multipart).",
    )
    object_type = serializers.CharField(
        help_text="Which object type to count in the image.",
    )
    use_finetuned_classifier = serializers.BooleanField(
        required=False,
        help_text="If true, force use finetuned classifier (when available). If false, force base. If omitted, auto.",
    )

    def _validate_resolution(self, uploaded):
        min_w = getattr(settings, "IMAGE_UPLOAD_MIN_WIDTH", 64)
        min_h = getattr(settings, "IMAGE_UPLOAD_MIN_HEIGHT", 64)
        max_w = getattr(settings, "IMAGE_UPLOAD_MAX_WIDTH", 8192)
        max_h = getattr(settings, "IMAGE_UPLOAD_MAX_HEIGHT", 8192)
        # Try to preserve file pointer position
        pos = None
        try:
            pos = uploaded.tell()
        except Exception:
            pos = None
        try:
            img = Image.open(uploaded)
            img.verify()  # quick validation
            # Re-open to access size after verify()
            try:
                if pos is not None and hasattr(uploaded, "seek"):
                    uploaded.seek(pos)
            except Exception:
                pass
            img2 = Image.open(uploaded)
            width, height = img2.size
            # Rewind for downstream saving
            try:
                if hasattr(uploaded, "seek"):
                    uploaded.seek(0)
            except Exception:
                pass
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise serializers.ValidationError({
                "message": "Upload a valid image. The file appears invalid, too large, or corrupted.",
                "limits": {
                    "min_width": int(min_w),
                    "min_height": int(min_h),
                    "max_width": int(max_w),
                    "max_height": int(max_h),
                },
            })
        finally:
            try:
                if pos is not None and hasattr(uploaded, "seek"):
                    uploaded.seek(pos)
            except Exception:
                pass
        if width < min_w or height < min_h or width > max_w or height > max_h:
            raise serializers.ValidationError({
                "message": "Image resolution out of allowed range.",
                "limits": {
                    "min_width": int(min_w),
                    "min_height": int(min_h),
                    "max_width": int(max_w),
                    "max_height": int(max_h),
                },
                "actual": {
                    "width": int(width),
                    "height": int(height),
                },
            })
        return uploaded

    def validate_image(self, value):
        return self._validate_resolution(value)

    def validate_images(self, value):
        # Validate each image in the list; report the first offending index
        for idx, item in enumerate(value):
            try:
                self._validate_resolution(item)
            except serializers.ValidationError as e:
                detail = e.detail if hasattr(e, "detail") else {"message": str(e)}
                raise serializers.ValidationError({
                    "index": idx,
                    "error": detail,
                })
        return value

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
    # Accept 0–100 as percentages
    blur = serializers.IntegerField(min_value=0, max_value=100, default=0)
    rotate = serializers.ListField(
        child=serializers.IntegerField(min_value=0, max_value=360),
        min_length=1
    )
    noise = serializers.IntegerField(min_value=0, max_value=100, default=0)