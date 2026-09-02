"""Views. These call into services.py (the src bridge), never src directly."""
from __future__ import annotations

from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render

from . import services
from .forms import AnalyseForm, BatchUploadForm
from .models import Decision, Submission


def landing(request):
    return render(request, "detector/landing.html")


def analyse(request):
    if request.method == "POST":
        form = AnalyseForm(request.POST, request.FILES)
        if form.is_valid():
            text = form.cleaned_data.get("text") or ""
            language = form.cleaned_data.get("language") or "auto"
            upload = form.cleaned_data.get("file")
            source_filename = ""
            if upload:
                source_filename = upload.name
                # Scaffold: only .txt is decoded inline; pdf/docx extraction is
                # wired later (pypdf / python-docx are in requirements.txt).
                if upload.name.lower().endswith(".txt"):
                    try:
                        text = upload.read().decode("utf-8", errors="replace")
                    except Exception:  # noqa: BLE001
                        text = ""

            submission = Submission.objects.create(
                text=text,
                language=language,
                source_filename=source_filename,
                uploaded_by=request.user if request.user.is_authenticated else None,
            )

            result = services.analyse_text(text, language)
            Decision.objects.create(
                submission=submission,
                verdict=result["verdict"],
                confidence=result["confidence"],
                driving_head=result["driving_head"],
                stylometric_score=result["stylometric_score"],
                curvature_score=result["curvature_score"],
                semantic_score=result["semantic_score"],
                explanation=result["explanation"],
                model_version=result["model_version"],
            )
            return redirect("detector:result", submission_id=submission.id)
    else:
        form = AnalyseForm()
    return render(request, "detector/analyse.html", {"form": form})


def result(request, submission_id):
    submission = get_object_or_404(Submission, pk=submission_id)
    decision = getattr(submission, "decision", None)
    top_features = []
    if decision and isinstance(decision.explanation, dict):
        top_features = decision.explanation.get("top_features", [])
    return render(
        request,
        "detector/result.html",
        {"submission": submission, "decision": decision, "top_features": top_features},
    )


def batch(request):
    form = BatchUploadForm(request.POST or None, request.FILES or None)
    submitted = False
    if request.method == "POST" and form.is_valid():
        # TODO(phase-3 step-3.8): parse CSV and analyse each row via services.
        submitted = True
    rows = services.placeholder_batch_rows()
    return render(
        request,
        "detector/batch.html",
        {"form": form, "rows": rows, "submitted": submitted},
    )


def dashboard(request):
    verdict_counts = list(
        Decision.objects.values("verdict").annotate(n=Count("verdict")).order_by("verdict")
    )
    language_counts = list(
        Submission.objects.values("language").annotate(n=Count("language")).order_by("language")
    )
    recent = Submission.objects.select_related("decision").order_by("-uploaded_at")[:10]
    return render(
        request,
        "detector/dashboard.html",
        {
            "verdict_counts": verdict_counts,
            "language_counts": language_counts,
            "recent": recent,
            "total": Submission.objects.count(),
        },
    )


def about(request):
    return render(request, "detector/about.html")
