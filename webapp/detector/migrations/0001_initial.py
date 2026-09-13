"""Initial schema for Submission and Decision.

Hand-authored to match webapp/detector/models.py exactly (Django was not
installed at scaffold time). Running `python manage.py makemigrations detector`
after installing requirements should report "No changes detected".
"""
import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Submission",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("text", models.TextField(blank=True)),
                (
                    "language",
                    models.CharField(
                        choices=[
                            ("auto", "Auto-detect"),
                            ("en", "English"),
                            ("hi", "Hindi"),
                            ("te", "Telugu"),
                            ("cm", "Code-mixed"),
                        ],
                        default="auto",
                        max_length=8,
                    ),
                ),
                ("source_filename", models.CharField(blank=True, max_length=255)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                (
                    "uploaded_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="submissions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-uploaded_at"],
            },
        ),
        migrations.CreateModel(
            name="Decision",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "verdict",
                    models.CharField(
                        choices=[("HUMAN", "Human"), ("ABSTAIN", "Abstain"), ("MACHINE", "Machine")],
                        max_length=8,
                    ),
                ),
                ("confidence", models.FloatField()),
                ("driving_head", models.CharField(max_length=32)),
                ("stylometric_score", models.FloatField()),
                ("curvature_score", models.FloatField()),
                ("semantic_score", models.FloatField(blank=True, null=True)),
                ("explanation", models.JSONField(blank=True, default=dict)),
                ("model_version", models.CharField(max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "submission",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="decision",
                        to="detector.submission",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
