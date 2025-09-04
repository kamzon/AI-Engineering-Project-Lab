import logging
import os
import shutil
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
from django.conf import settings
from drf_spectacular.utils import extend_schema, OpenApiResponse, OpenApiExample
from records.models import Result
from .serializers import (
    ResultSerializer,
    CountRequestSerializer,
    CorrectionRequestSerializer,
)
from pipeline.model import SamSegmentationClassifier

logger = logging.getLogger(__name__)

class CountView(APIView):
    """Run segmentation and count objects of a specific type in an image."""

    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        summary="Count objects in an uploaded image",
        description=(
            "Uploads an image and runs the segmentation/counting pipeline for the given "
            "object_type. Returns the created Result with predicted_count and metadata."
        ),
        request=CountRequestSerializer,
        responses={
            201: ResultSerializer,
            400: OpenApiResponse(description="Missing or invalid parameters"),
            500: OpenApiResponse(description="Processing error in pipeline"),
        },
        tags=["Counting"],
        examples=[
            OpenApiExample(
                "Car counting example",
                description="Multipart form-data upload for counting cars.",
                value={"object_type": "car"},
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        # Validate input via serializer so schema matches implementation
        input_serializer = CountRequestSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = input_serializer.validated_data
        image = validated["image"]
        object_type = validated["object_type"]
        pipeline_run = SamSegmentationClassifier()

        # create record and save file to MEDIA_ROOT
        result = Result.objects.create(image=image, object_type=object_type, status="processing")
        try:
            # pipeline_run must accept path + type and return dict with predicted_count + meta
            print("image:", result.image.path)
            #print("object_type:", result.object_type)
            pipeline_run.image_path = result.image.path
            #pipeline_run.candidate_labels = [result.object_type]
            output = pipeline_run.run()
            print('output:', output)
            print("label_counts:", output.get("label_counts", {}).get(result.object_type))
            result.predicted_count = output.get("label_counts", {}).get(result.object_type)
            # Save panoptic image into MEDIA_ROOT and store URL in meta
            run_id = output.get("id")
            panoptic_path = output.get("panoptic_path")
            meta = {}
            if run_id and panoptic_path and os.path.exists(panoptic_path):
                rel_path = os.path.join("outputs", f"{run_id}.png")
                dest_path = (settings.MEDIA_ROOT / rel_path)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy(panoptic_path, dest_path)
                    panoptic_url = settings.MEDIA_URL + \
                        rel_path.replace(os.sep, "/")
                    meta.update(
                        {"run_id": run_id, "panoptic_url": panoptic_url})
                except Exception:
                    logger.exception("Failed to copy panoptic image to media.")
            # Merge any meta from pipeline if present
            meta.update(output.get("meta", {}))
            result.meta = meta
            result.status = "predicted"
            result.save()
        except Exception as e:
            logger.exception("Pipeline failed for Result id=%s", result.id)
            result.status = "failed"
            result.meta = {"error": str(e)}
            result.save()
            return Response({"detail": "processing error", "error": str(e)}, status=500)

        return Response(ResultSerializer(result).data, status=status.HTTP_201_CREATED)


class CorrectionView(APIView):
    """Submit a corrected count for a prior result."""

    @extend_schema(
        summary="Submit corrected count",
        description=(
            "Updates a Result with a user-provided corrected_count. Can accept JSON or "
            "form-data payloads."
        ),
        request=CorrectionRequestSerializer,
        responses={
            200: ResultSerializer,
            400: OpenApiResponse(description="Validation error"),
            404: OpenApiResponse(description="Result not found"),
        },
        tags=["Counting"],
        examples=[
            OpenApiExample(
                "Corrected count example",
                value={"result_id": 1, "corrected_count": 42},
                request_only=True,
            )
        ],
    )
    def post(self, request):
        input_serializer = CorrectionRequestSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = input_serializer.validated_data
        result_id = validated["result_id"]
        corrected_count = validated["corrected_count"]

        result = get_object_or_404(Result, id=result_id)
        result.corrected_count = corrected_count
        result.status = "corrected"
        result.save()

        return Response(ResultSerializer(result).data, status=200)
