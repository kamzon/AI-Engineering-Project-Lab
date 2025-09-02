import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from records.models import Result
from .serializers import ResultSerializer
from pipeline.model import SamSegmentationClassifier

logger = logging.getLogger(__name__)

class CountView(APIView):
    """
    Accepts multipart form: 'image' file + 'object_type' string.
    Creates Result (status=processing), runs pipeline (sync), updates result.
    """
    def post(self, request, *args, **kwargs):
        pipeline_run = SamSegmentationClassifier()
        image = request.FILES.get("image")
        object_type = request.data.get("object_type")
        if not image or not object_type:
            return Response({"detail": "image and object_type are required"}, status=400)

        # create record and save file to MEDIA_ROOT
        result = Result.objects.create(image=image, object_type=object_type, status="processing")
        try:
            # pipeline_run must accept path + type and return dict with predicted_count + meta
            print("image:", result.image.path)
            print("object_type:", result.object_type)
            pipeline_run.image_path = result.image.path
            pipeline_run.candidate_labels = [result.object_type]
            output = pipeline_run.run()
            print('output:', output)
            result.predicted_count = int(output.get("predicted_count", 0))
            result.meta = output.get("meta", {})
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
    """
    Accepts form-data or json: result_id + corrected_count.
    Updates the record.
    """
    def post(self, request, *args, **kwargs):
        result_id = request.data.get("result_id")
        corrected_count = request.data.get("corrected_count")
        if result_id is None or corrected_count is None:
            return Response({"detail": "result_id and corrected_count are required"}, status=400)

        result = get_object_or_404(Result, id=result_id)
        try:
            result.corrected_count = int(corrected_count)
            result.status = "corrected"
            result.save()
        except ValueError:
            return Response({"detail": "corrected_count must be an integer"}, status=400)

        return Response(ResultSerializer(result).data, status=200)
