"""Views. These call into services.py (the src bridge), never src directly."""
from __future__ import annotations

import csv
import io

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


#: CSV columns: `text` required, `language` and `filename` optional.
_BATCH_TEXT_COL = "text"
_BATCH_LANGUAGE_COL = "language"
_BATCH_FILENAME_COL = "filename"
_MAX_BATCH_ROWS = 200


def _parse_batch_csv(uploaded_file) -> tuple[list[dict], list[str]]:
    """Parse the uploaded CSV into analyse_batch()-ready rows; return (rows, errors)."""
    errors: list[str] = []
    try:
        raw = uploaded_file.read().decode("utf-8-sig", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return [], [f"could not read file: {exc}"]

    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None or _BATCH_TEXT_COL not in reader.fieldnames:
        return [], [f"CSV must have a '{_BATCH_TEXT_COL}' column; found {reader.fieldnames}"]

    rows = []
    for i, record in enumerate(reader, start=1):
        text = (record.get(_BATCH_TEXT_COL) or "").strip()
        if not text:
            errors.append(f"row {i}: empty '{_BATCH_TEXT_COL}', skipped")
            continue
        if len(rows) >= _MAX_BATCH_ROWS:
            errors.append(f"stopped at {_MAX_BATCH_ROWS} rows (CSV has more)")
            break
        rows.append({
            "row": i, "text": text,
            "language": (record.get(_BATCH_LANGUAGE_COL) or "auto").strip() or "auto",
            "filename": (record.get(_BATCH_FILENAME_COL) or "").strip(),
        })
    return rows, errors


def batch(request):
    form = BatchUploadForm(request.POST or None, request.FILES or None)
    submitted = False
    rows: list[dict] = []
    errors: list[str] = []
    if request.method == "POST" and form.is_valid():
        parsed_rows, errors = _parse_batch_csv(form.cleaned_data["csv_file"])
        submitted = True
        try:
            analysed = services.analyse_batch(parsed_rows)
        except Exception as exc:  # noqa: BLE001 - surface as a page error, not a 500
            errors.append(f"analysis failed: {exc}")
            analysed = []
        for parsed, out in zip(parsed_rows, analysed):
            result = out.get("result")
            if result is not None:
                submission = Submission.objects.create(
                    text=parsed["text"], language=out["language"],
                    source_filename=parsed["filename"],
                    uploaded_by=request.user if request.user.is_authenticated else None,
                )
                Decision.objects.create(
                    submission=submission, verdict=result["verdict"], confidence=result["confidence"],
                    driving_head=result["driving_head"], stylometric_score=result["stylometric_score"],
                    curvature_score=result["curvature_score"], semantic_score=result["semantic_score"],
                    explanation=result["explanation"], model_version=result["model_version"],
                )
            rows.append({
                "row": out["row"], "filename": out["filename"] or f"row {out['row']}",
                "language": out["language"], "verdict": out["verdict"],
                "confidence": out["confidence"], "driving_head": out["driving_head"],
            })
    return render(
        request,
        "detector/batch.html",
        {"form": form, "rows": rows, "submitted": submitted, "errors": errors},
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
