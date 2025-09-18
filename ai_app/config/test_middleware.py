from django.test import SimpleTestCase, override_settings
from django.http import HttpResponse
from django.test.client import RequestFactory
from .middleware import RestrictAPIAccessMiddleware, StaticTokenMiddleware


class MiddlewareTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    @override_settings(ENFORCE_FRONTEND_ORIGIN=True, FRONTEND_ALLOWED_ORIGINS=["http://allowed.com"])
    def test_restrict_api_access_allows_allowed_origin(self):
        def get_response(_):
            return HttpResponse("ok")
        mw = RestrictAPIAccessMiddleware(get_response)
        req = self.factory.get("/api/test", HTTP_ORIGIN="http://allowed.com")
        resp = mw(req)
        self.assertEqual(resp.status_code, 200)

    @override_settings(ENFORCE_FRONTEND_ORIGIN=True, FRONTEND_ALLOWED_ORIGINS=["http://allowed.com"])
    def test_restrict_api_access_blocks_disallowed_origin(self):
        def get_response(_):
            return HttpResponse("ok")
        mw = RestrictAPIAccessMiddleware(get_response)
        req = self.factory.get("/api/test", HTTP_ORIGIN="http://evil.com")
        resp = mw(req)
        self.assertEqual(resp.status_code, 403)

    @override_settings(API_STATIC_TOKEN="secret")
    def test_static_token_middleware_blocks_missing_token(self):
        def get_response(_):
            return HttpResponse("ok")
        mw = StaticTokenMiddleware(get_response)
        req = self.factory.get("/api/test")
        resp = mw(req)
        self.assertEqual(resp.status_code, 401)

    @override_settings(API_STATIC_TOKEN="secret")
    def test_static_token_middleware_allows_bearer_token(self):
        def get_response(_):
            return HttpResponse("ok")
        mw = StaticTokenMiddleware(get_response)
        req = self.factory.get("/api/test", HTTP_AUTHORIZATION="Bearer secret")
        resp = mw(req)
        self.assertEqual(resp.status_code, 200)

    @override_settings(API_STATIC_TOKEN="secret")
    def test_static_token_middleware_allows_header_token(self):
        def get_response(_):
            return HttpResponse("ok")
        mw = StaticTokenMiddleware(get_response)
        req = self.factory.get("/api/test", HTTP_X_API_TOKEN="secret")
        resp = mw(req)
        self.assertEqual(resp.status_code, 200)


