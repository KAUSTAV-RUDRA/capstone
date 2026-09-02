"""Named URL routes for the detector app."""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("analyse/", views.analyse, name="analyse"),
    path("result/<uuid:submission_id>/", views.result, name="result"),
    path("batch/", views.batch, name="batch"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("about/", views.about, name="about"),
]
