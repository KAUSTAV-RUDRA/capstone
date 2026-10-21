"""Tests for the back-translation attack in src/data/attack.py (fake translator, no model loads)."""
import json
import logging

from src.data import attack
from src.data.attack import (BT_ROUTES, attacked_id, backtranslate, ollama_vram_models, run_backtranslation)
from src.data.schema import ATTACK_TYPES


class FakeTranslator:
    """Tags each sentence with the direction so the round trip is visible and checkable."""

    name = "fake-mt"

    def __init__(self):
        self.calls = []

    def translate(self, sentences, src, tgt):
        self.calls.append((src, tgt, len(sentences)))
        return [f"{s}|{src[:3]}>{tgt[:3]}" for s in sentences]


def _row(i, lang, label=1):
    return {"id": f"{lang}_{i}", "text": f"First one. Second one.\n\nThird {i}.", "language": lang,
            "label": label, "length_words": 6, "split": "test", "attack_type": "clean"}


def test_pivots_follow_spec():
    assert BT_ROUTES["en"] == ("eng_Latn", "hin_Deva")
    assert BT_ROUTES["hi"] == ("hin_Deva", "eng_Latn")
    assert BT_ROUTES["te"] == ("tel_Telu", "eng_Latn")
    assert "cm" not in BT_ROUTES


def test_backtranslate_round_trip_direction_and_paragraphs():
    tr = FakeTranslator()
    out = backtranslate(["A b. C d.\n\nE f."], "hi", tr)[0]
    assert [c[:2] for c in tr.calls] == [("hin_Deva", "eng_Latn"), ("eng_Latn", "hin_Deva")]
    assert out.count("\n\n") == 1
    assert "|hin>eng|eng>hin" in out


def test_backtranslate_empty_translation_keeps_source():
    class Empty(FakeTranslator):
        def translate(self, sentences, src, tgt):
            return ["" for _ in sentences]

    assert backtranslate(["Keep me."], "en", Empty())[0] == "Keep me."


def test_backtranslate_rejects_cm():
    try:
        backtranslate(["x"], "cm", FakeTranslator())
    except ValueError as exc:
        assert "not applicable" in str(exc)
    else:
        raise AssertionError("cm must be rejected")


def test_run_skips_cm_logs_it_and_fills_fields(tmp_path, caplog):
    rows = [_row(i, lang) for lang in ("en", "hi", "te", "cm") for i in range(5)]
    out = tmp_path / "a.jsonl"
    with caplog.at_level(logging.INFO, logger="attack"):
        totals = run_backtranslation(rows, out, {}, 42, 0, "cpu", cap=(3, 3), translator=FakeTranslator())
    got = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert totals["rows"] == len(got) == 9 and totals["not_applicable"] == 3
    assert not any(r["language"] == "cm" for r in got)
    assert "attack not applicable" in caplog.text
    r = got[0]
    assert r["id"] == attacked_id(r["source_id"], "backtranslation") and r["id"].endswith("_bt")
    assert r["attack_type"] == "back_translation" and r["attack_type"] in ATTACK_TYPES
    assert r["attack_changed"] is True and r["length_ratio"] > 0 and r["bt_model"] == "fake-mt"


def test_run_is_resumable_and_respects_cap(tmp_path):
    rows = [_row(i, "en") for i in range(10)]
    out = tmp_path / "a.jsonl"
    assert run_backtranslation(rows, out, {}, 42, 0, "cpu", cap=(4, 4), translator=FakeTranslator())["rows"] == 4
    again = run_backtranslation(rows, out, {}, 42, 0, "cpu", cap=(4, 4), translator=FakeTranslator())
    assert again["rows"] == 0 and again["skipped"] == 4
    assert len(out.read_text(encoding="utf-8").splitlines()) == 4


def test_no_translator_loaded_when_nothing_to_do(tmp_path, monkeypatch):
    rows = [_row(0, "en")]
    out = tmp_path / "a.jsonl"
    run_backtranslation(rows, out, {}, 42, 0, "cpu", translator=FakeTranslator())
    monkeypatch.setattr(attack, "load_translator", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert run_backtranslation(rows, out, {}, 42, 0, "cpu")["skipped"] == 1


def test_fallback_to_nllb_is_logged(monkeypatch, caplog):
    class Boom:
        def __init__(self, *a, **k):
            raise ImportError("IndicTransToolkit missing")

    class Nllb:
        name = "nllb-fake"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(attack, "IndicTrans2Translator", Boom)
    monkeypatch.setattr(attack, "NllbTranslator", Nllb)
    with caplog.at_level(logging.INFO, logger="attack"):
        tr = attack.load_translator({}, "cpu")
    assert tr.name == "nllb-fake"
    assert "falling back" in caplog.text and "nllb-fake" in caplog.text


def test_ollama_vram_detection(monkeypatch):
    import io
    import urllib.request

    body = json.dumps({"models": [{"name": "qwen2.5:7b-instruct", "size_vram": 5_000_000_000},
                                  {"name": "cpu-only", "size_vram": 0}]})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body.encode()))
    assert ollama_vram_models() == ["qwen2.5:7b-instruct"]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError))
    assert ollama_vram_models() == []
