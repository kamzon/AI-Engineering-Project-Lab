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
    finetuned_types = _get_finetuned_object_types() or []
    # Initial dropdown shows defaults; JS can swap to finetuned list when toggled
    object_types = list(default_object_types)
    background_types = ["random", "solid", "noise"]
    latest = Result.objects.order_by("-created_at").first()
    latest_is_unsafe = False
    if latest:
        try:
            meta = latest.meta
            if isinstance(meta, dict):
                latest_is_unsafe = latest.status in ("unsafe", "rejected") or bool(meta.get("unsafe") is True or meta.get("pred_label") == "unsafe")
            else:
                # Fallback: search for 'unsafe' in stringified meta
                latest_is_unsafe = latest.status in ("unsafe", "rejected") or ("unsafe" in str(meta).lower())
        except Exception:
            latest_is_unsafe = latest.status in ("unsafe", "rejected")
    pref = ThemePreference.objects.first()
    current_theme = pref.theme if pref else "winter"
    # Ensure CSRF token present for JS fetch usage
    get_token(request)
    return render(
        request,
        "index.html",
        {
            "latest": latest,
            "latest_is_unsafe": latest_is_unsafe,
            "object_types": object_types,
            "default_object_types": default_object_types,
            "finetuned_object_types": finetuned_types,
            "background_types": background_types,
            "current_theme": current_theme,
            "finetuned_available": bool(os.path.isdir(ModelConstants.FINETUNED_MODEL_DIR) and os.listdir(ModelConstants.FINETUNED_MODEL_DIR)),
        },
    )

def history(request):
    # Return all results as JSON for the "Show history" button
    results = Result.objects.order_by("-created_at")
    data = []
    for r in results:
        # Robust unsafe detection for corrections_allowed
        is_unsafe = False
        try:
            meta = r.meta
            if isinstance(meta, dict):
                is_unsafe = r.status in ("unsafe", "rejected") or bool(meta.get("unsafe") is True or meta.get("pred_label") == "unsafe")
            else:
                is_unsafe = r.status in ("unsafe", "rejected") or ("unsafe" in str(meta).lower())
        except Exception:
            is_unsafe = r.status in ("unsafe", "rejected")
        data.append({
            "id": r.id,
            "object_type": r.object_type,
            "status": r.status,
            "predicted_count": r.predicted_count,
            "corrected_count": r.corrected_count,
            "created_at": r.created_at.isoformat(),
            "image": r.image.url,
            "meta": r.meta,
            "corrections_allowed": not is_unsafe,
        })
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