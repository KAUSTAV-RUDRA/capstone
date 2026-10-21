"""Tests for the back-translation attack in src/data/attack.py (fake translator, no model loads)."""
import json
import logging

from src.data import attack
from src.data.attack import (BT_ROUTES, attacked_id, backtranslate, checked_backtranslation, ollama_vram_models,
                            run_backtranslation, script_ok)
from src.data.schema import ATTACK_TYPES


SAMPLE = {"eng_Latn": "Hello there friend.", "hin_Deva": "नमस्ते मेरे दोस्त।", "tel_Telu": "హలో నా మిత్రమా."}
PARA = chr(10) * 2
LANG_TEXT = {"en": "First one. Second one." + PARA + "Third.",
             "hi": "पहला वाक्य। दूसरा वाक्य।" + PARA + "तीसरा।",
             "te": "మొదటి వాక్యం. రెండవ వాక్యం." + PARA + "మూడవది.",
             "cm": "yeh ek test hai. aur doosra."}


class FakeTranslator:
    """Returns text in the target script, so the script check passes; records every call."""

    name = "fake-mt"

    def __init__(self):
        self.calls = []

    def translate(self, sentences, src, tgt, greedy=False):
        self.calls.append((src, tgt, len(sentences), greedy))
        return [SAMPLE[tgt] for _ in sentences]


def _row(i, lang, label=1):
    return {"id": f"{lang}_{i}", "text": LANG_TEXT[lang], "language": lang,
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
    assert out.startswith(SAMPLE["hin_Deva"])


def test_backtranslate_empty_translation_keeps_source():
    class Empty(FakeTranslator):
        def translate(self, sentences, src, tgt, greedy=False):
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
        tr = attack.load_translator({"attack": {"backtranslation": {"backend": "indictrans2"}}}, "cpu")
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


def test_script_ok_thresholds():
    assert script_ok("Hello world", "en") and not script_ok("Hello в в в в в в в в в в", "en")
    assert script_ok("नमस्ते मेरे प्यारे दुनिया hi", "hi") and not script_ok("hello world नमस्ते", "hi")
    assert script_ok("హలో ప్రపంచం", "te") and not script_ok("", "te")


class Flaky(FakeTranslator):
    """Wrong script on beam search, right script on greedy (or never right)."""

    def __init__(self, fix_on_greedy):
        super().__init__()
        self.fix = fix_on_greedy

    def translate(self, sentences, src, tgt, greedy=False):
        self.calls.append((src, tgt, len(sentences), greedy))
        if greedy and self.fix:
            return [SAMPLE[tgt] for _ in sentences]
        return ["в в в в" for _ in sentences]


def test_failed_check_retries_greedy_then_succeeds():
    tr = Flaky(fix_on_greedy=True)
    (text, failed), = checked_backtranslation(["Some source."], "en", tr)
    assert not failed and script_ok(text, "en")
    assert [c[3] for c in tr.calls] == [False, False, True, True]


def test_failed_after_greedy_keeps_source_and_flags(tmp_path, caplog):
    rows = [_row(0, "te")]
    out = tmp_path / "a.jsonl"
    with caplog.at_level(logging.INFO, logger="attack"):
        totals = run_backtranslation(rows, out, {}, 42, 0, "cpu", translator=Flaky(fix_on_greedy=False))
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["bt_failed"] is True and r["attack_changed"] is False and r["text"] == LANG_TEXT["te"]
    assert totals["failed"] == 1 and "'te': 1" in caplog.text
