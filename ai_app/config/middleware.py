from django.conf import settings
from django.http import HttpResponseForbidden


class RestrictAPIAccessMiddleware:
    """
    Deny requests to API endpoints unless the Origin/Referer matches configured frontend origins.

    Applies to paths starting with settings.API_PATH_PREFIX (default '/api/').
    Checks the 'Origin' header first, then falls back to 'Referer'.
    If ENFORCE_FRONTEND_ORIGIN is False, the middleware does nothing.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.prefix = getattr(settings, "API_PATH_PREFIX", "/api/")
        self.allowed = set(getattr(settings, "FRONTEND_ALLOWED_ORIGINS", []))
        self.enabled = bool(getattr(settings, "ENFORCE_FRONTEND_ORIGIN", True))

    def __call__(self, request):
        if self.enabled and request.path.startswith(self.prefix):
            origin = request.headers.get("Origin")
            referer = request.headers.get("Referer")

            def normalize(url: str | None) -> str | None:
                if not url:
                    return None
                # Keep scheme + host + optional port
                try:
                    from urllib.parse import urlsplit

                    parts = urlsplit(url)
                    base = f"{parts.scheme}://{parts.netloc}"
                    return base.rstrip('/')
                except Exception:
                    return None

            source = normalize(origin) or normalize(referer)
            if self.allowed and source not in self.allowed:
                return HttpResponseForbidden("Forbidden: origin not allowed")

        return self.get_response(request)
