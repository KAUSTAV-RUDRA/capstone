"""Webapp tests. Real inference (mGPT/Head A) is mocked at services.analyse_text
so these run fast and offline; language_id / CSV parsing are pure logic and
run for real.
"""
from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from .models import Decision, Submission
from .views import _parse_batch_csv


def _fake_result(verdict: str = "ABSTAIN") -> dict:
    return {
        "verdict": verdict, "confidence": 0.7, "language": "en", "driving_head": "stylometric",
        "stylometric_score": 0.4, "curvature_score": 0.3, "semantic_score": None,
        "explanation": {"top_features": [], "note": "x", "is_decision_support": True,
                        "language": "en", "bucket": "en"},
        "model_version": "v1.0-qwen7b",
    }


class PageSmokeTests(TestCase):
    def test_landing_about_analyse_batch_dashboard_get_200(self) -> None:
        for name in ("detector:landing", "detector:about", "detector:analyse",
                     "detector:batch", "detector:dashboard"):
            resp = self.client.get(reverse(name))
            self.assertEqual(resp.status_code, 200, name)


class AnalyseFlowTests(TestCase):
    @patch("webapp.detector.services.analyse_text")
    def test_analyse_post_creates_submission_and_decision_and_redirects(self, mock_analyse) -> None:
        mock_analyse.return_value = _fake_result("MACHINE")
        resp = self.client.post(reverse("detector:analyse"),
                                {"text": "hello world, this is a test submission.", "language": "auto"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Submission.objects.count(), 1)
        decision = Decision.objects.get()
        self.assertEqual(decision.verdict, "MACHINE")
        mock_analyse.assert_called_once()

    def test_analyse_post_without_text_or_file_is_rejected(self) -> None:
        resp = self.client.post(reverse("detector:analyse"), {"language": "auto"})
        self.assertEqual(resp.status_code, 200)  # re-renders the form with an error
        self.assertEqual(Submission.objects.count(), 0)

    @patch("webapp.detector.services.analyse_text")
    def test_result_page_shows_decision(self, mock_analyse) -> None:
        mock_analyse.return_value = _fake_result("HUMAN")
        resp = self.client.post(reverse("detector:analyse"), {"text": "some text here", "language": "auto"})
        submission = Submission.objects.get()
        resp = self.client.get(reverse("detector:result", args=[submission.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "HUMAN")


class DashboardAggregationTests(TestCase):
    @patch("webapp.detector.services.analyse_text")
    def test_dashboard_counts_reflect_real_decisions(self, mock_analyse) -> None:
        mock_analyse.side_effect = [_fake_result("MACHINE"), _fake_result("HUMAN"), _fake_result("MACHINE")]
        for _ in range(3):
            self.client.post(reverse("detector:analyse"), {"text": "text " * 5, "language": "auto"})
        resp = self.client.get(reverse("detector:dashboard"))
        self.assertEqual(resp.status_code, 200)
        counts = {row["verdict"]: row["n"] for row in resp.context["verdict_counts"]}
        self.assertEqual(counts.get("MACHINE"), 2)
        self.assertEqual(counts.get("HUMAN"), 1)
        self.assertEqual(resp.context["total"], 3)


class BatchCsvParsingTests(TestCase):
    def test_parses_text_language_filename_columns(self) -> None:
        from io import BytesIO

        csv_bytes = b"text,language,filename\nhello world,en,a.txt\nnamaste duniya,hi,b.txt\n"
        rows, errors = _parse_batch_csv(BytesIO(csv_bytes))
        self.assertEqual(len(rows), 2)
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["text"], "hello world")
        self.assertEqual(rows[0]["language"], "en")
        self.assertEqual(rows[1]["filename"], "b.txt")

    def test_missing_text_column_is_an_error(self) -> None:
        from io import BytesIO

        rows, errors = _parse_batch_csv(BytesIO(b"foo,bar\n1,2\n"))
        self.assertEqual(rows, [])
        self.assertTrue(errors)

    def test_empty_text_rows_are_skipped_with_a_note(self) -> None:
        from io import BytesIO

        rows, errors = _parse_batch_csv(BytesIO(b"text\nhello\n\n  \n"))
        self.assertEqual(len(rows), 1)
        self.assertTrue(any("empty" in e for e in errors))

    @patch("webapp.detector.services.analyse_text")
    def test_batch_post_persists_decisions_for_each_row(self, mock_analyse) -> None:
        from io import BytesIO

        from django.core.files.uploadedfile import SimpleUploadedFile

        mock_analyse.return_value = _fake_result("ABSTAIN")
        csv_file = SimpleUploadedFile("rows.csv", b"text\nfirst row\nsecond row\n", content_type="text/csv")
        resp = self.client.post(reverse("detector:batch"), {"csv_file": csv_file})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Submission.objects.count(), 2)
        self.assertEqual(mock_analyse.call_count, 2)


class LanguageIdentifierTests(TestCase):
    def test_resolve_language_honours_explicit_choice_over_detection(self) -> None:
        from . import services

        # A Devanagari-script text explicitly forced to "en" should stay "en".
        assert services._resolve_language("यह हिंदी है", "en") == "en"
        assert services._resolve_language("यह हिंदी है", "auto") == "hi"
        assert services._resolve_language("plain english text here", None) == "en"
