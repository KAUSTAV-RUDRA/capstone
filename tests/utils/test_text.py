from src.utils.text import collapse_whitespace, normalise_rows


def test_collapse_whitespace_newlines_runs_and_unicode_space():
    assert collapse_whitespace(" a\n\nb \t c d \n") == "a b c d"


def test_normalise_rows_only_touches_text():
    rows = [{"id": "x", "text": "a\nb", "label": 1}]
    out = normalise_rows(rows)
    assert out == [{"id": "x", "text": "a b", "label": 1}]
    assert rows[0]["text"] == "a\nb"          # input not mutated
