"""Data models: Submission and its Decision (the audit trail).

The ORM is the decision-evidence trail for consultancy (locked decision §5).
NON-NEGOTIABLE #8: a Decision is decision support, never an automatic verdict.
"""
from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

LANGUAGE_CHOICES = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("hi", "Hindi"),
    ("te", "Telugu"),
    ("cm", "Code-mixed"),
]

VERDICT_CHOICES = [
    ("HUMAN", "Human"),
    ("ABSTAIN", "Abstain"),
    ("MACHINE", "Machine"),
]


class Submission(models.Model):
    """A single piece of text submitted for analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    text = models.TextField(blank=True)
    language = models.CharField(max_length=8, choices=LANGUAGE_CHOICES, default="auto")
    source_filename = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="submissions",
    )

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"Submission {self.id} ({self.language})"


class Decision(models.Model):
    """The system's decision-support output for one Submission."""

    submission = models.OneToOneField(
        Submission,
        on_delete=models.CASCADE,
        related_name="decision",
    )
    verdict = models.CharField(max_length=8, choices=VERDICT_CHOICES)
    confidence = models.FloatField()
    driving_head = models.CharField(max_length=32)
    stylometric_score = models.FloatField()
    curvature_score = models.FloatField()
    semantic_score = models.FloatField(null=True, blank=True)
    explanation = models.JSONField(default=dict, blank=True)
    model_version = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Decision {self.verdict} for {self.submission_id}"
