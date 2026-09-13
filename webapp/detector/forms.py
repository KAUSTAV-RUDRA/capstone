"""Forms for single-text analysis and batch CSV upload (CSRF on by default)."""
from __future__ import annotations

from django import forms

from .models import LANGUAGE_CHOICES


class AnalyseForm(forms.Form):
    """Single-submission form: paste text OR upload a txt/pdf/docx file."""

    text = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 10, "class": "form-control",
                                     "placeholder": "Paste student text here..."}),
        label="Text",
    )
    file = forms.FileField(
        required=False,
        widget=forms.ClearableFileInput(attrs={"class": "form-control",
                                               "accept": ".txt,.pdf,.docx"}),
        label="or upload a file (.txt / .pdf / .docx)",
    )
    language = forms.ChoiceField(
        choices=LANGUAGE_CHOICES,
        initial="auto",
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Language",
    )

    def clean(self) -> dict:
        cleaned = super().clean()
        if not cleaned.get("text") and not cleaned.get("file"):
            raise forms.ValidationError("Provide some text or upload a file.")
        return cleaned


class BatchUploadForm(forms.Form):
    """Batch form: upload a CSV of submissions."""

    csv_file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={"class": "form-control",
                                               "accept": ".csv"}),
        label="CSV file",
    )
