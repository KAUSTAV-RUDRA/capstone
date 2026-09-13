"""Admin registration - the free audit interface for consultancy (locked §5)."""
from __future__ import annotations

from django.contrib import admin

from .models import Decision, Submission


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    list_display = ("id", "language", "source_filename", "uploaded_by", "uploaded_at")
    list_filter = ("language", "uploaded_at")
    search_fields = ("id", "source_filename", "text")
    readonly_fields = ("id", "uploaded_at")
    date_hierarchy = "uploaded_at"


@admin.register(Decision)
class DecisionAdmin(admin.ModelAdmin):
    list_display = (
        "submission",
        "verdict",
        "confidence",
        "driving_head",
        "model_version",
        "created_at",
    )
    list_filter = ("verdict", "submission__language", "driving_head", "created_at")
    search_fields = ("submission__id",)
    readonly_fields = ("created_at",)
