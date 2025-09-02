from django.contrib import admin
from .models import Result

@admin.register(Result)
class ResultAdmin(admin.ModelAdmin):
    list_display = ("id", "object_type", "predicted_count", "corrected_count", "status", "created_at")
    list_filter = ("status", "object_type")
    search_fields = ("object_type",)
    readonly_fields = ("created_at", "updated_at")