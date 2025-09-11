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
from pipeline.models.few_shot import FewShotResNet
from pipeline.config import ModelConstants
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
    """Generate images, then fine-tune the ResNet classifier using few-shot on them."""

    @extend_schema(
        summary="Generate image with specified objects",
        description=(
            "Generates images with specified objects, saves them into a labeled dataset, "
            "then fine-tunes the ResNet image classifier using few-shot learning."
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
        print("input_serializer is valid")
        v = input_serializer.validated_data
        num_images = v["num_images"]
        num_objects_per_image = v["max_objects_per_image"]
        object_types = list(v.get("selected_type", []))  # Ensure a fresh copy is used
        backgrounds = list(v.get("backgrounds", []))  # Ensure a fresh copy is used
        blur = v.get("blur", 0)
        rotate_choices = v["rotate"]
        noise = v["noise"]
        results = []
        # Build dataset mapping: class_name -> [image_path, ...]
        class_to_image_paths = {}
        # Save under MEDIA_ROOT/fewshot_dataset/<class>/image.png
        dataset_root = (settings.MEDIA_ROOT / "fewshot_dataset")
        dataset_root.mkdir(parents=True, exist_ok=True)
        print("dataset_root is created")
        for _ in range(num_images):
            background = random.choice(backgrounds)
            rotate = random.choice(rotate_choices)
            if num_objects_per_image == 1:
                prompt = f"A {background} background with a single {object_types[0]}"
            else:
                prompt = f"A {background} background with {num_objects_per_image} objects of a {object_types[0]}s"
            print("prompt is created")
            print(prompt)
            try:
                print("generating image")
                img = generate_image_with_api(prompt, api_key=API_KEY)
                print("image generated")
                #print(img)
                img = augment_image(img, blur=blur, rotate=rotate, noise=noise)
                print("img is created")
                print(img)
                print("object_types is created")
                print(object_types[0])
                if img is None or not object_types[0]:
                    results.append({"error": "Image generation failed or no objects."})
                    continue
                # Choose one class label per image for few-shot supervision
                labeled_class = object_types[0]
                class_dir = dataset_root / labeled_class
                class_dir.mkdir(parents=True, exist_ok=True)
                print("class_dir is created")
                filename = f"{labeled_class}_{random.randint(100000,999999)}.png"
                img_path = class_dir / filename
                img.save(img_path, format='PNG')
                class_to_image_paths.setdefault(labeled_class, []).append(str(img_path))
                print("class_to_image_paths is created")
                print(class_to_image_paths)
                results.append({
                    "saved": True,
                    "class": labeled_class,
                    "path": str(img_path)
                })
            except Exception as e:
                print("error is created")
                print(e)
                results.append({"error": str(e)})
        # After dataset is built, run few-shot fine-tuning and persist the model
        try:
            fewshot = FewShotResNet(
                lr=ModelConstants.FEW_SHOT_LR,
                weight_decay=ModelConstants.FEW_SHOT_WEIGHT_DECAY,
                max_epochs=ModelConstants.FEW_SHOT_MAX_EPOCHS,
                batch_size=ModelConstants.FEW_SHOT_BATCH_SIZE,
                freeze_backbone=ModelConstants.FEW_SHOT_FREEZE_BACKBONE,
            )
            print("fewshot is created")
            print(fewshot)
            fewshot.finetune(class_to_image_paths)
            finetuned_dir = ModelConstants.FINETUNED_MODEL_DIR
        except Exception as e:
            print("error is created")
            print(e)
            return Response({
                "results": results,
                "message": "Dataset created but fine-tuning failed.",
                "error": str(e)
            }, status=500)

        return Response({
            "results": results,
            "classes": sorted(class_to_image_paths.keys()),
            "images_per_class": {k: len(v) for k, v in class_to_image_paths.items()},
            "finetuned_model_dir": finetuned_dir
        }, status=201)
        
        
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