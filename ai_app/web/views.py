from django.shortcuts import render
from django.conf import settings
from records.models import Result
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.middleware.csrf import get_token
from .models import ThemePreference
import os
import json
from pipeline.config import ModelConstants

def _get_finetuned_object_types():
    try:
        finetuned_dir = ModelConstants.FINETUNED_MODEL_DIR
        if os.path.isdir(finetuned_dir) and os.listdir(finetuned_dir):
            config_path = os.path.join(finetuned_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                id2label = cfg.get("id2label")
                if isinstance(id2label, dict) and id2label:
                    # Sort by numeric id to keep stable ordering
                    ordered = [label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
                    return ordered 
    except Exception:
        pass
    return None


def index(request):
    default_object_types = getattr(settings, "OBJECT_TYPES", [])
    finetuned_types = _get_finetuned_object_types()
    object_types = finetuned_types or default_object_types
    background_types = ["random", "solid", "noise"]
    latest = Result.objects.order_by("-created_at").first()
    pref = ThemePreference.objects.first()
    current_theme = pref.theme if pref else "winter"
    # Ensure CSRF token present for JS fetch usage
    get_token(request)
    return render(
        request,
        "index.html",
        {
            "latest": latest,
            "object_types": object_types,
            "background_types": background_types,
            "current_theme": current_theme,
        },
    )

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
            "meta": r.meta,
        }
        for r in results
    ]
    return JsonResponse(data, safe=False)


@require_POST
def set_theme(request):
    theme = request.POST.get("theme") or (request.headers.get("Content-Type", "").startswith("application/json") and (request.body or b""))
    if isinstance(theme, (bytes, bytearray)):
        import json
        try:
            payload = json.loads(theme.decode("utf-8"))
            theme = payload.get("theme")
        except Exception:
            return HttpResponseBadRequest("Invalid JSON")
    allowed = {"dracula", "winter", "forest", "dim", "night", "halloween"}
    if theme not in allowed:
        return HttpResponseBadRequest("Invalid theme")
    pref, _ = ThemePreference.objects.get_or_create(id=1)
    pref.theme = theme
    pref.save()
    return JsonResponse({"theme": pref.theme})