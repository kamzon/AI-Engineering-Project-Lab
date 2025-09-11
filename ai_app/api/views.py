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
from pipeline.pipeline import Pipeline
from .metrics import record_pipeline_metrics
import sys
import random
import io
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from image_generation import generate_image_with_api, augment_image, OBJECT_TYPES, BACKGROUND_TYPES, API_KEY, API_UPLOAD_ENDPOINT, API_CORRECT_ENDPOINT
# --- Generation API Endpoint ---
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser


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
        input_serializer = CountRequestSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = input_serializer.validated_data
        object_type = validated["object_type"]
        single_image = validated.get("image")
        images_list = validated.get("images")

        def process_one(uploaded_image):
            pipeline_run = Pipeline()
            res = Result.objects.create(
                image=uploaded_image, object_type=object_type, status="processing")
            try:
                pipeline_run.image_path = res.image.path
                pipeline_run.candidate_labels = [res.object_type, 'other']
                output = pipeline_run.run()
                try:
                    record_pipeline_metrics(
                        output.get("metadata", {}), output.get(
                            "label_counts", {})
                    )
                except Exception:
                    logger.exception("Failed to record Prometheus metrics")
                res.predicted_count = output.get(
                    "label_counts", {}).get(res.object_type)
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


class GenerateView(APIView):
    """Generate an image with specified objects using an external API."""

    @extend_schema(
        summary="Generate image with specified objects",
        description=(
            "Generates an image containing specified objects using an external image "
            "generation API. Returns the created Result with a link to the generated image."
        ),
        request=CorrectionRequestSerializer,  
        responses={
            201: ResultSerializer,
            400: OpenApiResponse(description="Missing or invalid parameters"),
            500: OpenApiResponse(description="Image generation error"),
        },
        tags=["Generation"],
        examples=[
            OpenApiExample(
                "Image generation example",
                description="Request to generate an image with specified objects.",
                value={"object_type": "cat", "num_objects": 3},
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        input_serializer = CorrectionRequestSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = input_serializer.validated_data
        object_type = validated["object_type"]
        num_objects = validated.get("num_objects", 1)

        generated_image_path = f"generated_images/{object_type}_{num_objects}.png"
        os.makedirs(os.path.dirname(generated_image_path), exist_ok=True)
        with open(generated_image_path, 'wb') as f:
            f.write(os.urandom(1024))  

        result = Result.objects.create(
            image=generated_image_path,
            object_type=object_type,
            predicted_count=num_objects,
            status="generated",
            meta={"generation_method": "external_api"}
        )

        return Response(ResultSerializer(result).data, status=status.HTTP_201_CREATED)
    
class GenerateView(APIView):
    """Generate an image with specified objects using the real image generation script."""

    @extend_schema(
        summary="Generate image with specified objects",
        description=(
            "Generates an image containing specified objects using the real image generation script. "
            "Returns the created Result with a link to the generated image."
        ),
        request=".serializers.GenerationRequestSerializer",
        responses={
            201: ResultSerializer,
            400: OpenApiResponse(description="Missing or invalid parameters"),
            500: OpenApiResponse(description="Image generation error"),
        },
        tags=["Generation"],
        examples=[
            OpenApiExample(
                "Image generation example",
                description="Request to generate an image with specified objects.",
                value={"object_type": "cat", "num_objects": 3},
                request_only=True,
            ),
        ],
    )
    def post(self, request):
        from .serializers import GenerationRequestSerializer
        input_serializer = GenerationRequestSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=400)
        v = input_serializer.validated_data
        num_images = v["num_images"]
        max_objects_per_image = v["max_objects_per_image"]
        object_types = v["object_types"]
        backgrounds = v["backgrounds"]
        blur = v["blur"]
        rotate_choices = v["rotate"]
        noise = v["noise"]
        results = []
        for _ in range(num_images):
            num_objects = random.randint(1, max_objects_per_image)
            chosen_types = random.choices(object_types, k=num_objects)
            background = random.choice(backgrounds)
            rotate = random.choice(rotate_choices)
            prompt = f"A {background} background with " + ", ".join(chosen_types)
            try:
                img = generate_image_with_api(prompt, api_key=API_KEY)
                img = augment_image(img, blur=blur, rotate=rotate, noise=noise)
                if img is None or not chosen_types:
                    results.append({"error": "Image generation failed or no objects."})
                    continue
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                # For perfect correction: always send the correct count for the posted object_type
                posted_object_type = chosen_types[0]
                correct_count = sum(1 for t in chosen_types if t == posted_object_type)
                files = {'image': ('test.png', buf, 'image/png')}
                data = {'object_type': posted_object_type}
                upload_resp = requests.post(API_UPLOAD_ENDPOINT, files=files, data=data)
                if not upload_resp.ok:
                    results.append({"error": upload_resp.text})
                    continue
                result = upload_resp.json()
                correction_data = {
                    "result_id": result["id"],
                    "corrected_count": correct_count
                }
                corr_resp = requests.post(API_CORRECT_ENDPOINT, data=correction_data)
                results.append({
                    "result": result,
                    "correction_status": corr_resp.status_code,
                    "correction_response": corr_resp.text,
                    "correction_sent": {"object_type": posted_object_type, "corrected_count": correct_count}
                })
            except Exception as e:
                results.append({"error": str(e)})
        return Response({"results": results}, status=201)
        
        
@api_view(["POST"])
@permission_classes([IsAdminUser])
def generate_and_upload_image(request):
    """
    Generate a random image using the AI endpoint, upload it to the pipeline, and return the result.
    """
    num_objects = random.randint(1, 3)
    chosen_types = random.choices(OBJECT_TYPES, k=num_objects)
    background = random.choice(BACKGROUND_TYPES)
    blur = 0
    rotate = random.choice([0, 90, 180, 270])
    noise = 0

    prompt = f"A {background} background with " + ", ".join(chosen_types)
    try:
        img = generate_image_with_api(prompt, api_key=API_KEY)
        img = augment_image(img, blur=blur, rotate=rotate, noise=noise)
        selected_object = chosen_types[0]
        correct_count = chosen_types.count(selected_object)
        if img is None:
            return Response({"error": "Image generation failed."}, status=500)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        files = {'image': ('test.png', buf, 'image/png')}
        data = {'object_type': selected_object}
        upload_resp = requests.post(API_UPLOAD_ENDPOINT, files=files, data=data)
        if not upload_resp.ok:
            return Response({"error": upload_resp.text}, status=500)
        result = upload_resp.json()
        correction_data = {
            "result_id": result["id"],
            "corrected_count": correct_count
        }
        corr_resp = requests.post(API_CORRECT_ENDPOINT, data=correction_data)
        return Response({
            "result": result,
            "correction_status": corr_resp.status_code,
            "correction_response": corr_resp.text
        }, status=201)
    except Exception as e:
        return Response({"error": str(e)}, status=500)