"""App configuration for the detector app."""
from __future__ import annotations

from django.apps import AppConfig


class DetectorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "webapp.detector"
    label = "detector"
    verbose_name = "MGT Detector"
