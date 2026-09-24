"""Tests for src.data.clean_artifacts (scaffold stub). Serves Phase 2 §2.1.4."""
from __future__ import annotations

from src.data.clean_artifacts import strip_boilerplate

#: Verbatim from gemma's 2026-09-22 cm gate probe: the two rows that failed the
#: emoji check, both with the emoji inside a trailing offer to continue.
GEMMA_POSTAMBLE_1 = ("Arey yaar, Shawn Levy ne direction ki hai movie ka! Steel story Richard "
                     "Matheson ne likhi thi na, yeh bahut interesting lag rahi hai..\n\n"
                     "Let me know if you'd like me to continue the conversation! \U0001F60A")
GEMMA_POSTAMBLE_2 = ("Court acche decisions deti hai par yeh BJP sarkar bilkul tanashahi mein chal "
                     "rahi hai yaar. Samvida karmchariyon ka koi respect nahin hota is government "
                     "ke saamne.\n\nLet me know if you want me to write more! \U0001F60A")

#: Also verbatim from that probe: the same words used INSIDE the passage. A
#: substring rule strips these and eats text the model was asked to write.
GEMMA_IN_DIALOGUE_1 = ("Arre nahi! It's Friday night, let's make it happen. You know kya film "
                       "dekhne ka mood hai? Let me know!")
GEMMA_IN_DIALOGUE_2 = ("My knowledge of Hindi slang is limited at best. I trust your judgement "
                       "more than Google Translate on this one.")


def test_strips_the_trailing_offer_line_and_its_emoji() -> None:
    for text in (GEMMA_POSTAMBLE_1, GEMMA_POSTAMBLE_2):
        out = strip_boilerplate(text)
        assert "Let me know" not in out and "\U0001F60A" not in out, out
        assert out.rstrip().endswith((".", "!", "?")), out      # the passage itself survives intact
    assert strip_boilerplate(GEMMA_POSTAMBLE_1).startswith("Arey yaar, Shawn Levy")
    # Other sign-off shapes, each as a whole final line.
    for line in ("I hope this helps!", "Hope that helps.", "Feel free to ask if you need more!",
                 "Would you like me to write another one?", "Do you want me to continue?",
                 "I can also write a longer version if you want."):
        assert strip_boilerplate(f"Yeh mera passage hai yaar.\n\n{line}") == "Yeh mera passage hai yaar."


def test_leaves_the_same_words_alone_inside_the_passage() -> None:
    # The other direction, and the reason the rule is anchored to a full final line.
    for text in (GEMMA_IN_DIALOGUE_1, GEMMA_IN_DIALOGUE_2):
        assert strip_boilerplate(text) == text
    # A final line that merely ENDS with the phrase is passage text, not packaging.
    tail = "Kal milte hain, aur agar plan change hua toh let me know if you like."
    assert strip_boilerplate(f"Aaj movie dekhne ja rahe hain.\n{tail}").endswith(tail)
    # Mid-passage chatter is left to the preamble/refusal rules, not this one.
    middle = "Hope this helps!\nAb asli baat yeh hai ki exam kal hai."
    assert strip_boilerplate(middle) == middle


def test_strips_stacked_sign_offs_but_never_the_whole_passage() -> None:
    stacked = "Yeh passage hai.\n\nWould you like me to continue?\nLet me know if you want more!"
    assert strip_boilerplate(stacked) == "Yeh passage hai."
    # A row that is nothing but chatter empties out rather than looping past the cap;
    # the emoji and length checks then see it for what it is.
    assert strip_boilerplate("Let me know if you'd like me to continue!") == ""
    assert strip_boilerplate("") == ""


def test_strips_preambles_and_reports_fraction() -> None:
    """'Sure! Here is...' preambles removed; fraction stripped reported."""
    from src.data.clean_artifacts import clean_text

    out, frac = clean_text("Sure! Here is the passage:\nRivers flow downhill.", "en")
    assert out == "Rivers flow downhill."
    assert 0.4 < frac < 0.6
    assert clean_text("Rivers flow downhill.", "en") == ("Rivers flow downhill.", 0.0)


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except NotImplementedError:
                print(f"TODO {name}")          # scaffold placeholder, not a failure
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)


def test_clean_strips_preamble_and_markdown_but_keeps_in_character_sure() -> None:
    from src.data.clean_artifacts import clean_text

    text = "Certainly! Here is a short passage of about 150 words on the topic:\n\n**Stars** burn hydrogen."
    assert clean_text(text, "en")[0] == "Stars burn hydrogen."
    assert clean_text("Certainly! A stack is a data structure.", "en")[0] == "A stack is a data structure."
    assert clean_text("Sure, buddy! Abhi main khel rha hoon.", "cm")[0].startswith("Sure, buddy!")


def test_drop_rules_are_symmetric_and_judge_cleaned_text() -> None:
    from src.data.clean_artifacts import clean_rows

    rows = [
        {"id": "h", "label": 0, "language": "te", "text": "తెలుగు వార్త ఇది.", "source": "x"},
        {"id": "m1", "label": 1, "language": "te", "text": "తెలుగు व वार्त.", "generator": "g"},
        {"id": "m2", "label": 1, "language": "cm", "text": "Yeh accha hai 😊 yaar.", "generator": "g"},
        {"id": "m3", "label": 1, "language": "hi", "text": "कुछ पाठ <|im_start|>user", "generator": "g"},
        {"id": "m4", "label": 1, "language": "en", "text": "Fine text.", "generator": "g", "truncated": True},
        {"id": "m5", "label": 1, "language": "cm", "text": "Accha hai.\n\nLet me know if you'd like more! 😊",
         "generator": "g"},
    ]
    kept, dropped, _ = clean_rows(rows, enforce_guard=False)
    reasons = {r["id"]: r["drop_reasons"] for r in dropped}
    assert reasons == {"m1": ["stray_script"], "m2": ["emoji"], "m3": ["echo"], "m4": ["truncated"]}
    assert {r["id"] for r in kept} == {"h", "m5"}          # m5's emoji was in stripped chatter
