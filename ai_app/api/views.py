import logging
import os
import shutil
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404
import json as _json
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
            pipeline_run = Pipeline(use_finetuned_classifier=validated.get("use_finetuned_classifier"))
            res = Result.objects.create(
                image=uploaded_image, object_type=object_type, status="processing")
            try:
                pipeline_run.image_path = res.image.path
                pipeline_run.candidate_labels = [res.object_type, 'other']
                output = pipeline_run.run()
                # If the pipeline aborted due to safety (or any error), mark as unsafe and stop early
                if output.get("error"):
                    # Preserve default predicted_count (0) and mark unsafe
                    res.predicted_count = 0
                    meta = {"error": output.get("error"), "unsafe": True, "pred_label": "unsafe"}
                    # Merge any metadata the pipeline may have returned
                    meta.update(output.get("metadata", {}))
                    res.meta = meta
                    res.status = "unsafe"
                    res.save()
                    # Record metrics even for unsafe outcomes if metadata is present
                    try:
                        record_pipeline_metrics(
                            output.get("metadata", {}), output.get("label_counts", {})
                        )
                    except Exception:
                        logger.exception("Failed to record Prometheus metrics for unsafe outcome")
                    return res
                try:
                    record_pipeline_metrics(
                        output.get("metadata", {}), output.get(
                            "label_counts", {})
                    )
                except Exception:
                    logger.exception("Failed to record Prometheus metrics")
                # Only assign predicted_count if present; otherwise keep default (0)
                count_map = output.get("label_counts") or {}
                count_val = count_map.get(res.object_type)
                if count_val is not None:
                    res.predicted_count = count_val
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
                # Keep predicted_count at default (0) to satisfy NOT NULL constraints
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
        # Disallow corrections for unsafe results (handle dict or JSON string in meta)
        meta = result.meta
        if isinstance(meta, str):
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        # Block corrections if the result was flagged unsafe/failed/processing/rejected, or meta indicates unsafe
        meta_is_dict = isinstance(meta, dict)
        meta_pred_label_unsafe = meta_is_dict and (meta.get("pred_label") == "unsafe")
        meta_has_reason_unsafe = meta_is_dict and isinstance(meta.get("reason"), dict) and (
            meta.get("reason", {}).get("pred_label") == "unsafe" or meta.get("reason", {}).get("unsafe") is True
        )
        meta_error_unsafe = meta_is_dict and isinstance(meta.get("error"), str) and ("unsafe" in meta.get("error").lower())
        meta_flag_unsafe = meta_is_dict and (meta.get("unsafe") is True)
        if result.status in ("unsafe", "failed", "processing", "rejected") or meta_flag_unsafe or meta_pred_label_unsafe or meta_has_reason_unsafe or meta_error_unsafe:
            return Response({
                "detail": "Corrections are not allowed for unsafe images.",
                "id": result.id,
                "status": result.status
            }, status=status.HTTP_400_BAD_REQUEST)
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
        object_types = list(v.get("object_types", []))  # Ensure a fresh copy is used
        backgrounds = list(v.get("backgrounds", []))  # Ensure a fresh copy is used
        # Frontend sends blur/noise as 0–100; map to augmentation-friendly values
        blur_pct = int(v.get("blur", 0))
        rotate_choices = v["rotate"]
        noise_pct = int(v["noise"]) if v.get("noise") is not None else 0
        results = []
        # Save under MEDIA_ROOT/fewshot_dataset/<class>/image.png and MEDIA_ROOT/fewshot_dataset/others/image.png
        dataset_root = (settings.MEDIA_ROOT / "fewshot_dataset")
        (dataset_root / "others").mkdir(parents=True, exist_ok=True)
        (dataset_root / object_types[0]).mkdir(parents=True, exist_ok=True)

        positive_paths = []
        negative_paths = []

        for _ in range(num_images):
            background = random.choice(backgrounds)
            rotate = random.choice(rotate_choices)
            # Build one positive sample prompt that includes target
            pos_prompt = f"A {background} background with {num_objects_per_image} {object_types[0]}"
            try:
                img = generate_image_with_api(pos_prompt, api_key=API_KEY)
                blur_radius = round((max(0, min(100, blur_pct)) / 100.0) * 10.0, 2)
                noise_std = round((max(0, min(100, noise_pct)) / 100.0) * 50.0, 2)
                img = augment_image(img, blur=blur_radius, rotate=rotate, noise=noise_std)
                if img is None:
                    results.append({"error": "Positive image generation failed."})
                else:
                    pos_dir = dataset_root / object_types[0]
                    filename = f"pos_{random.randint(100000,999999)}.png"
                    img_path = pos_dir / filename
                    img.save(img_path, format='PNG')
                    positive_paths.append(str(img_path))
                    results.append({"saved": True, "class": object_types[0], "path": str(img_path)})
            except Exception as e:
                results.append({"error": str(e)})

            # Build one negative sample prompt that avoids the target (sample from defaults)
            neg_candidates = [o for o in ModelConstants.DEFAULT_CANDIDATE_LABELS if o != object_types[0]]
            chosen = random.sample(neg_candidates, k=min(num_objects_per_image, len(neg_candidates)))
            neg_prompt = f"A {background} background with " + ", ".join(chosen)
            try:
                img = generate_image_with_api(neg_prompt, api_key=API_KEY)
                img = augment_image(img, blur=blur_radius, rotate=rotate, noise=noise_std)
                if img is None:
                    results.append({"error": "Negative image generation failed."})
                else:
                    neg_dir = dataset_root / "others"
                    filename = f"neg_{random.randint(100000,999999)}.png"
                    img_path = neg_dir / filename
                    img.save(img_path, format='PNG')
                    negative_paths.append(str(img_path))
                    results.append({"saved": True, "class": "others", "path": str(img_path)})
            except Exception as e:
                results.append({"error": str(e)})

        # After dataset is built, run few-shot fine-tuning and persist the model
        try:
            image_paths = positive_paths + negative_paths
            if not image_paths:
                return Response({"results": results, "message": "No images generated."}, status=500)
            trainer = FewShotResNet(object_type=object_types[0])
            trainer.load_data_from_paths(image_paths=image_paths, test_size=0.2)
            trainer.finetune(
                num_train_epochs=ModelConstants.FEW_SHOT_MAX_EPOCHS,
                per_device_batch_size=ModelConstants.FEW_SHOT_BATCH_SIZE,
            )
            finetuned_dir = ModelConstants.FINETUNED_MODEL_DIR
        except Exception as e:
            return Response({
                "results": results,
                "message": "Dataset created but fine-tuning failed.",
                "error": str(e)
            }, status=500)

        return Response({
            "results": results,
            "positives": len(positive_paths),
            "negatives": len(negative_paths),
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