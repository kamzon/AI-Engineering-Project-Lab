from django.db import models


class ThemePreference(models.Model):
    """Stores a single global theme preference for the site."""
    THEME_CHOICES = [
        ("winter", "winter"),
        ("dracula", "dracula"),
        ("forest", "forest"),
        ("dim", "dim"),
        ("night", "night"),
        ("halloween", "halloween"),
    ]

    theme = models.CharField(max_length=20, choices=THEME_CHOICES, default="winter")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"ThemePreference(theme={self.theme})"
