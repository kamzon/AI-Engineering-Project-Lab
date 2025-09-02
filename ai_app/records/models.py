from django.db import models

class Result(models.Model):
    image = models.ImageField(upload_to="uploads/%Y/%m/%d/")
    object_type = models.CharField(max_length=64)
    predicted_count = models.IntegerField(default=0)
    corrected_count = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=16, default="pending")  # pending/predicted/corrected
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.object_type} | id={self.id}"
