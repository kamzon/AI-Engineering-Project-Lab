from django.shortcuts import render
from django.conf import settings
from records.models import Result
from django.http import JsonResponse

def index(request):
    object_types = getattr(settings, "OBJECT_TYPES", [])
    latest = Result.objects.order_by("-created_at").first()
    return render(request, "index.html", {"latest": latest, "object_types": object_types})

def history(request):
    # Return all results as JSON for the "Show history" button
    results = Result.objects.order_by("-created_at")
    data = [
        {
            "id": r.id,
            "object_type": r.object_type,
            "status": r.status,
            "predicted_count": r.predicted_count,
            "corrected_count": r.corrected_count,
            "created_at": r.created_at.isoformat(),
            "image": r.image.url,
        }
        for r in results
    ]
    return JsonResponse(data, safe=False)