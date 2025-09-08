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
from .metrics import record_pipeline_metrics

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
        object_type = validated["object_type"]
        single_image = validated.get("image")
        images_list = validated.get("images")

        def process_one(uploaded_image):
            pipeline_run = SamSegmentationClassifier()
            res = Result.objects.create(
                image=uploaded_image, object_type=object_type, status="processing")
            try:
                pipeline_run.image_path = res.image.path
                output = pipeline_run.run()
                # Record Prometheus metrics using metadata and label counts
                try:
                    record_pipeline_metrics(
                        output.get("metadata", {}), output.get(
                            "label_counts", {})
                    )
                except Exception:
                    logger.exception("Failed to record Prometheus metrics")
                res.predicted_count = output.get(
                    "label_counts", {}).get(res.object_type)
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
                        logger.exception(
                            "Failed to copy panoptic image to media.")
                # Merge pipeline metadata into the stored meta
                meta.update(output.get("metadata", {}))
                res.meta = meta
                res.status = "predicted"
                res.save()
            except Exception as e:
                logger.exception("Pipeline failed for Result id=%s", res.id)
                res.status = "failed"
                res.meta = {"error": str(e)}
                res.save()
            return res

        if images_list:
            results = [process_one(img) for img in images_list]
            data = ResultSerializer(results, many=True).data
            return Response(data, status=status.HTTP_201_CREATED)
        else:
            result = process_one(single_image)
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
