"""Offline tests for src.data.build_human_corpus (no network).

Run with pytest, or directly:  python tests/data/test_build_human_corpus.py
Serves docs/parts-plan.md Part 1.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.data import build_human_corpus as bhc


def test_detokenise_ptb() -> None:
    raw = 'Basically there are " Best Seller " lists , and you \'re not wrong ( IIRC ) . It does n\'t matter URL_0 .'
    out = bhc.detokenise_ptb(raw)
    assert out == 'Basically there are "Best Seller" lists, and you\'re not wrong (IIRC). It doesn\'t matter.'
    assert bhc.detokenise_ptb("see here ( URL_0 ) for details .") == "see here for details."


def test_repair_holes_and_artefact_check() -> None:
    assert bhc.repair_holes("Albedo (; ) is the fraction of sunlight.") == "Albedo is the fraction of sunlight."
    assert bhc.repair_holes("Lincoln ( ; February 12, 1809 - April 15, 1865) was") == "Lincoln (February 12, 1809 - April 15, 1865) was"
    assert bhc.has_artefact("Algeria covers an area of , making it the largest")   # unrepairable template hole
    assert bhc.has_artefact("the state s {\\displaystyle s} and the decision")     # LaTeX from a Wikipedia scrape
    assert bhc.has_artefact("see [[link]] here") and bhc.has_artefact("bad byte � here")
    assert not bhc.has_artefact("wait ... what? Fine (really). Done.")             # spaced ellipsis is fine


def test_fix_sentence_spacing() -> None:
    assert bhc.fix_sentence_spacing("lasted until 449 BC.In 494 BC, the U.S. Army") == "lasted until 449 BC. In 494 BC, the U.S. Army"


def test_wikipedia_prose_drops_headings_and_lists() -> None:
    text = "Lead sentence one. Lead sentence two.\n\nEtymology and definition\n\nBody paragraph here.\n* list item\n| table"
    assert bhc.wikipedia_prose(text) == "Lead sentence one. Lead sentence two. Body paragraph here."


def test_passage_from_doc_bounds_and_determinism() -> None:
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(80)]  # 7 words each
    doc = " ".join(sentences)
    p1 = bhc.passage_from_doc(doc, 120, 300, key="doc-a")
    p2 = bhc.passage_from_doc(doc, 120, 300, key="doc-a")
    assert p1 == p2 and p1 is not None
    assert 120 <= bhc.n_words(p1) <= 300
    assert p1.endswith(".")  # cut at a sentence boundary
    assert bhc.passage_from_doc(" ".join(sentences[:10]), 120, 300, key="short") is None
    lens = {bhc.n_words(bhc.passage_from_doc(doc, 120, 300, key=f"k{i}")) for i in range(20)}
    assert len(lens) > 3, "target length should vary per document"


def test_samanantar_sentence_filter() -> None:
    ok = bhc.samanantar_sentence_ok
    assert ok("The court has fixed a hearing for February 12 in the case.")
    assert not ok("Mithali To Anchor Indian Team Against Australia in ODIs")  # headline
    assert not ok("Jharkhand chief minister Hemant Soren")                    # too short
    assert not ok("its the same case here and there")                         # lowercase start
    assert not ok('"Jesus responded, ""How is it that the scribes say that?"')  # broken quoting
    assert not ok("Alia Bhatt , Sidharth Malhotra and Fawad Khan attended the event.")  # tokenised fragment
    assert not ok("The revised rates for the year 2016-17 are indicated below :-")     # trailing separator
    assert not ok("यह हिंदी वाक्य है और यह अंग्रेज़ी नहीं है ठीक")           # not English side


def test_allocate_sums_to_target() -> None:
    assert bhc.allocate(2200, {"general": 0.5, "indian": 0.5}) == {"general": 1100, "indian": 1100}
    alloc = bhc.allocate(2201, {"a": 1, "b": 1, "c": 1})
    assert sum(alloc.values()) == 2201 and max(alloc.values()) - min(alloc.values()) <= 1


def test_deduper_flags_near_duplicates() -> None:
    d = bhc.Deduper(num_perm=128, threshold=0.8, shingle_words=5)
    base = " ".join(f"word{i}" for i in range(200))
    assert d.check_and_add("a", base)
    assert not d.check_and_add("b", base + " extra tail")          # near-copy
    assert d.check_and_add("c", " ".join(f"other{i}" for i in range(200)))


def _fake_loader(n_docs: int):
    def loader(spec, ctx):
        for i in range(n_docs):
            text = " ".join(f"Doc {i} sentence {j} carries some distinct token t{i}x{j} here." for j in range(40))
            yield bhc.RawDoc(source=spec["name"], source_id=str(i), text=text, domain="fake", provenance="test")
    return loader


def test_build_is_resumable_and_idempotent() -> None:
    bhc.SOURCE_LOADERS["fake"] = _fake_loader(30)
    with tempfile.TemporaryDirectory() as tmp:
        config = {"human_corpus": {
            "output_dir": tmp, "passage_words": {"min": 120, "max": 300},
            "dedup": {"num_perm": 128, "threshold": 0.8, "shingle_words": 5},
            "buckets": {"en": {"bands": {
                "general": {"share": 0.5, "sources": [{"name": "fake", "max_share": 1.0}]},
                "indian": {"share": 0.5, "sources": [{"name": "fake", "max_share": 1.0}]},
            }}},
        }}
        s1 = bhc.build("en", 20, config)
        assert s1["rows"] == 20 and s1["bands"]["general"]["rows"] == 10 and s1["bands"]["indian"]["rows"] == 10
        # Second run: nothing new to pull, file unchanged, ids unique.
        s2 = bhc.build("en", 20, config)
        rows = [json.loads(l) for l in Path(tmp, "en.jsonl").read_text(encoding="utf-8").splitlines()]
        assert s2["rows"] == 20 and len({r["id"] for r in rows}) == 20
        assert all(120 <= r["length_words"] <= 300 and r["label"] == 0 and r["language"] == "en" for r in rows)
        # Raising the target resumes: only the shortfall is added.
        s3 = bhc.build("en", 30, config)
        assert s3["rows"] == 30 and s3["bands"]["general"]["rows"] == 15
    del bhc.SOURCE_LOADERS["fake"]


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
