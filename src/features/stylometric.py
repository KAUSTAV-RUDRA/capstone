"""Head A - stylometric feature extraction (docs/project-context-master.md §3).

Features: function words, burstiness, punctuation, TTR/MTLD, POS n-grams,
syntactic depth. Language-aware feature norms per bucket.

Implements the shared detector contract ``score(texts) -> np.ndarray`` and the
extractor contract ``fit`` / ``transform`` / ``feature_names``.

Serves docs/master-execution-plan.md Phase 1 (Day 2) and Phase 2 §2.2.2.

Design note (Phase 1 scope)
---------------------------
Every feature here is computed with the standard library + numpy only. No
spaCy/stanza model is loaded, so extraction never triggers a model download and
never crashes on Devanagari or Telugu input (non-negotiable: multilingual stays
in scope). Word tokenisation is Unicode-aware; sentence splitting understands the
Devanagari danda (``।``).

The docstring lists "POS n-grams, syntactic depth". True POS-tag n-grams and
dependency-tree depth need a per-bucket tagger/parser and are deferred to Phase 2
§2.2.2 (they plug in behind this same interface). Until then they are realised as
script-agnostic proxies that carry the same stylistic signal without a model:

- POS n-grams        -> function/content-word transition ratios (a word is
                        "function" iff it is in the bucket's function-word list).
- syntactic depth    -> punctuation-delimited clause count per sentence.

``score`` is a transparent, *provisional* unsupervised head (human text is
burstier / less repetitive than machine text). The trained fusion head that
replaces it is P1's Phase-2 work; this keeps the shared contract callable for the
Day-4 Django demo without pretending to be a calibrated verdict.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# --- Language-aware function-word lists (one per bucket) ---------------------
# Small, high-frequency closed-class sets. Enough for a stable function-word
# ratio and function/content transition proxies; not meant to be exhaustive.
_FUNCTION_WORDS_EN = {
    "the", "of", "and", "a", "to", "in", "is", "it", "that", "for", "on", "with",
    "as", "was", "at", "by", "an", "be", "this", "are", "from", "or", "but",
    "not", "they", "he", "she", "we", "you", "i", "his", "her", "its", "their",
    "our", "your", "my", "have", "has", "had", "will", "would", "can", "could",
    "should", "do", "does", "did", "so", "if", "then", "than", "which", "who",
    "what", "when", "where", "why", "how", "all", "any", "some", "no", "nor",
    "been", "being", "am", "were", "more", "most", "such", "only", "own", "same",
    "too", "very", "just", "also", "there", "here", "into", "over", "after",
    "before", "about", "up", "out", "off", "down", "because", "while", "these",
    "those",
}
_FUNCTION_WORDS_HI = {
    "के", "का", "की", "को", "में", "से", "है", "हैं", "और", "पर", "यह", "वह",
    "कि", "तो", "ने", "एक", "हो", "था", "थी", "थे", "कर", "नहीं", "भी", "जो",
    "ही", "अपने", "इस", "उस", "कुछ", "सब", "लिए", "साथ", "तक", "बाद", "पहले",
    "या", "अगर", "लेकिन", "क्या", "कैसे", "जब", "तब", "यहाँ", "वहाँ", "गया",
    "गई", "रहे", "रहा", "रही", "हुआ", "हुई", "करते", "जाती", "जाते",
}
_FUNCTION_WORDS_TE = {
    "అని", "ఒక", "మరియు", "ఈ", "ఆ", "ఇది", "అది", "కు", "లో", "నుండి", "తో",
    "కూడా", "కాదు", "ఉంది", "ఉన్న", "గా", "పై", "వరకు", "అయితే", "కానీ",
    "ఎందుకంటే", "ఏమి", "ఎలా", "ఎప్పుడు", "వారు", "నేను", "మేము", "మీరు", "కి",
    "ను", "ది", "లు", "పండుగలను", "చాలా", "వివిధ", "ఇక్కడ", "అక్కడ",
}
# Code-mixed (Hinglish, Latin script): romanised Hindi function words + English.
_FUNCTION_WORDS_CM = {
    "hai", "ka", "ki", "ko", "ke", "me", "mein", "se", "aur", "par", "ye", "wo",
    "kya", "to", "na", "ek", "ho", "tha", "thi", "the", "nahi", "nahin", "bhi",
    "jo", "hi", "apne", "is", "us", "kuch", "sab", "liye", "tum", "main", "hum",
    "aap", "kar", "bahut", "matlab", "yaar", "phir", "gaya", "gayi", "raha",
    "rahe", "hua", "hui",
} | _FUNCTION_WORDS_EN

_FUNCTION_WORDS = {
    "en": _FUNCTION_WORDS_EN,
    "hi": _FUNCTION_WORDS_HI,
    "te": _FUNCTION_WORDS_TE,
    "cm": _FUNCTION_WORDS_CM,
}

# Sentence terminators: Latin + Devanagari danda / double danda + newline.
_SENT_SPLIT_RE = re.compile(r"[.!?।॥\n]+")
# Unicode-aware word token (letters/marks/digits across scripts).
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
# Clause-boundary punctuation used as a script-agnostic syntactic-depth proxy.
_CLAUSE_PUNCT = ",;:—–"

# Ordered feature schema. transform() columns follow this exactly.
_FEATURE_NAMES: list[str] = [
    # lexical diversity
    "ttr",
    "root_ttr",
    "mtld",
    "hapax_ratio",
    "bigram_repeat_ratio",
    # length / shape
    "mean_word_len",
    "std_word_len",
    "mean_sent_len",
    "std_sent_len",
    "sent_len_burstiness",
    "word_len_burstiness",
    # function-word / POS-pattern proxies
    "function_word_ratio",
    "func_func_bigram_ratio",
    "content_content_bigram_ratio",
    # syntactic-depth proxy
    "mean_clauses_per_sent",
    "max_clauses_per_sent",
    # punctuation
    "punct_ratio",
    "comma_per_word",
    "period_per_sent",
    "question_ratio",
    "exclaim_ratio",
    "quote_ratio",
    "punct_diversity",
    # character classes
    "digit_ratio",
    "uppercase_ratio",
]


def _burstiness(values: list[float]) -> float:
    """Coefficient-of-variation burstiness ``(σ-μ)/(σ+μ)`` in ``[-1, 1]``.

    Human writing is burstier (higher) than machine writing; ``0`` when there is
    too little to measure.
    """
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mu = float(arr.mean())
    sigma = float(arr.std())
    denom = sigma + mu
    if denom == 0.0:
        return 0.0
    return (sigma - mu) / denom


def _mtld_one_direction(tokens: list[str], threshold: float = 0.72) -> float:
    """One-directional MTLD factor count (McCarthy & Jarvis, 2010)."""
    if not tokens:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    length = 0
    for tok in tokens:
        length += 1
        types.add(tok)
        if len(types) / length <= threshold:
            factors += 1.0
            types = set()
            length = 0
    if length > 0:  # trailing partial factor
        ttr = len(types) / length
        factors += (1.0 - ttr) / (1.0 - threshold)
    if factors == 0.0:
        return float(len(tokens))
    return len(tokens) / factors


def _mtld(tokens: list[str]) -> float:
    """Bidirectional MTLD (mean of forward + backward passes)."""
    if len(tokens) < 10:  # unstable on very short text; report raw diversity
        return float(len(set(tokens)))
    return 0.5 * (_mtld_one_direction(tokens) + _mtld_one_direction(tokens[::-1]))


class StylometricExtractor:
    """Extracts and (optionally) scores stylometric features."""

    def __init__(self, language: str | None = None, config: dict | None = None) -> None:
        """Args:
        language: Optional bucket for language-aware norms (``en``/``hi``/``te``/``cm``).
        config: Loaded ``configs/default.yaml``.
        """
        self.language = language
        self.config = config or {}
        self._feature_names = list(_FEATURE_NAMES)
        # Standardise transform() output once fitted (per "feature norms" in the
        # plan). Configurable; on by default so downstream ML sees zero-mean data.
        self.standardize: bool = bool(self.config.get("standardize", True))
        # Fitted state (populated by fit()).
        self.fitted_: bool = False
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.bucket_norms_: dict[str, dict[str, np.ndarray]] = {}

    # -- language routing ----------------------------------------------------
    def _bucket_for(self, text: str) -> str:
        """Pick the function-word bucket: explicit ``language`` else script guess.

        Script guess covers hi/te; Latin defaults to ``en`` (cm cannot be told
        from en by script alone and uses the en-superset list anyway).
        """
        if self.language in _FUNCTION_WORDS:
            return self.language  # type: ignore[return-value]
        for ch in text:
            code = ord(ch)
            if 0x0900 <= code <= 0x097F:
                return "hi"
            if 0x0C00 <= code <= 0x0C7F:
                return "te"
        return "en"

    # -- per-text feature computation ---------------------------------------
    def _features_for_text(self, text: str) -> list[float]:
        bucket = self._bucket_for(text)
        func_words = _FUNCTION_WORDS[bucket]

        words = _WORD_RE.findall(text)
        words_lower = [w.lower() for w in words]
        n_words = len(words)
        sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        n_sents = max(len(sentences), 1)
        total_chars = max(len(text), 1)

        # --- lexical diversity ---
        types = set(words_lower)
        n_types = len(types)
        ttr = n_types / n_words if n_words else 0.0
        root_ttr = n_types / math.sqrt(n_words) if n_words else 0.0
        mtld = _mtld(words_lower)
        counts = Counter(words_lower)
        hapax = sum(1 for c in counts.values() if c == 1)
        hapax_ratio = hapax / n_words if n_words else 0.0
        bigrams = list(zip(words_lower, words_lower[1:]))
        if bigrams:
            bg_counts = Counter(bigrams)
            repeated = sum(c for c in bg_counts.values() if c > 1)
            bigram_repeat_ratio = repeated / len(bigrams)
        else:
            bigram_repeat_ratio = 0.0

        # --- length / shape ---
        word_lens = [len(w) for w in words]
        mean_word_len = float(np.mean(word_lens)) if word_lens else 0.0
        std_word_len = float(np.std(word_lens)) if word_lens else 0.0
        sent_word_counts = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
        mean_sent_len = float(np.mean(sent_word_counts))
        std_sent_len = float(np.std(sent_word_counts))
        sent_len_burstiness = _burstiness([float(x) for x in sent_word_counts])
        word_len_burstiness = _burstiness([float(x) for x in word_lens])

        # --- function-word / POS-pattern proxies ---
        is_func = [w in func_words for w in words_lower]
        function_word_ratio = sum(is_func) / n_words if n_words else 0.0
        if len(is_func) >= 2:
            func_pairs = list(zip(is_func, is_func[1:]))
            n_pairs = len(func_pairs)
            func_func_bigram_ratio = sum(a and b for a, b in func_pairs) / n_pairs
            content_content_bigram_ratio = (
                sum((not a) and (not b) for a, b in func_pairs) / n_pairs
            )
        else:
            func_func_bigram_ratio = 0.0
            content_content_bigram_ratio = 0.0

        # --- syntactic-depth proxy: punctuation-delimited clauses per sentence ---
        clause_counts = [
            1 + sum(s.count(p) for p in _CLAUSE_PUNCT) for s in sentences
        ] or [0]
        mean_clauses_per_sent = float(np.mean(clause_counts))
        max_clauses_per_sent = float(np.max(clause_counts))

        # --- punctuation ---
        punct_chars = [ch for ch in text if not ch.isalnum() and not ch.isspace()]
        n_punct = len(punct_chars)
        punct_ratio = n_punct / total_chars
        comma_per_word = text.count(",") / n_words if n_words else 0.0
        period_per_sent = text.count(".") / n_sents
        question_ratio = text.count("?") / n_sents
        exclaim_ratio = text.count("!") / n_sents
        quote_ratio = sum(text.count(q) for q in "\"'“”‘’") / total_chars
        punct_diversity = len(set(punct_chars)) / n_punct if n_punct else 0.0

        # --- character classes ---
        digit_ratio = sum(ch.isdigit() for ch in text) / total_chars
        alpha = [ch for ch in text if ch.isalpha()]
        uppercase_ratio = (
            sum(ch.isupper() for ch in alpha) / len(alpha) if alpha else 0.0
        )

        return [
            ttr, root_ttr, mtld, hapax_ratio, bigram_repeat_ratio,
            mean_word_len, std_word_len, mean_sent_len, std_sent_len,
            sent_len_burstiness, word_len_burstiness,
            function_word_ratio, func_func_bigram_ratio,
            content_content_bigram_ratio,
            mean_clauses_per_sent, max_clauses_per_sent,
            punct_ratio, comma_per_word, period_per_sent, question_ratio,
            exclaim_ratio, quote_ratio, punct_diversity,
            digit_ratio, uppercase_ratio,
        ]

    def _raw_matrix(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._feature_names)), dtype=float)
        rows = [self._features_for_text(t) for t in texts]
        return np.asarray(rows, dtype=float)

    # -- public API ----------------------------------------------------------
    def fit(self, texts: list[str], languages: list[str] | None = None) -> "StylometricExtractor":
        """Fit feature normalisers/vocabularies on training texts.

        Stores global mean/std for standardisation, plus per-bucket norms (used
        for feature-importance analysis, plan §2.2.2). Vocabularies are the fixed
        curated function-word lists, so nothing is learned there.

        Args:
            texts: Training texts.
            languages: Optional parallel list of buckets for per-language norms.

        Returns:
            self.
        """
        raw = self._raw_matrix(texts)
        if raw.shape[0] == 0:
            raise ValueError("fit() requires at least one text.")
        self.mean_ = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std == 0.0] = 1.0  # guard constant features against divide-by-zero
        self.std_ = std

        if languages is not None:
            if len(languages) != len(texts):
                raise ValueError("languages must be parallel to texts.")
            for bucket in sorted(set(languages)):
                idx = [i for i, b in enumerate(languages) if b == bucket]
                sub = raw[idx]
                b_std = sub.std(axis=0)
                b_std[b_std == 0.0] = 1.0
                self.bucket_norms_[bucket] = {"mean": sub.mean(axis=0), "std": b_std}

        self.fitted_ = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return a feature matrix of shape ``(len(texts), n_features)``.

        Raw features before :meth:`fit`; globally standardised after (when
        ``standardize`` is on), so the schema/shape is identical either way.
        """
        raw = self._raw_matrix(texts)
        if self.fitted_ and self.standardize and raw.shape[0] > 0:
            return (raw - self.mean_) / self.std_
        return raw

    def fit_transform(self, texts: list[str], languages: list[str] | None = None) -> np.ndarray:
        """Convenience: :meth:`fit` then :meth:`transform`."""
        return self.fit(texts, languages).transform(texts)

    def feature_names(self) -> list[str]:
        """Return the ordered feature names matching :meth:`transform` columns."""
        return list(self._feature_names)

    def score(self, texts: list[str]) -> np.ndarray:
        """Return per-text machine-likelihood scores in ``[0, 1]`` (higher = more
        machine-like).

        PROVISIONAL unsupervised head. Grounded in the documented finding that
        human text is burstier and less repetitive than machine text, it maps a
        few raw features through a logistic squash. It is a stand-in for the
        trained fusion head (Phase 2, P1) and must never be shown as a calibrated
        verdict (non-negotiable #8: decision support, never accusation).
        """
        raw = self._raw_matrix(texts)
        if raw.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        cols = {name: i for i, name in enumerate(self._feature_names)}
        sent_burst = raw[:, cols["sent_len_burstiness"]]
        word_burst = raw[:, cols["word_len_burstiness"]]
        repeat = raw[:, cols["bigram_repeat_ratio"]]
        # Less bursty + more repetitive -> more machine-like. Weights are modest
        # and provisional; the trained head replaces them.
        logit = 2.0 * (-sent_burst) + 1.0 * (-word_burst) + 3.0 * repeat
        return 1.0 / (1.0 + np.exp(-logit))


if __name__ == "__main__":
    # Day-2 smoke test: 5 samples (en x2, hi, te, hinglish). Must not crash on
    # Devanagari/Telugu. Prints feature-matrix shape and ordered feature names.
    samples = [
        # en x2
        "The quick brown fox jumps over the lazy dog. It was a bright cold day "
        "in April, and the clocks were striking thirteen.",
        "Education is the most powerful tool for change. When students learn to "
        "think critically, they question assumptions, weigh evidence, and reach "
        "their own conclusions.",
        # hi (Devanagari)
        "भारत एक विशाल देश है। यहाँ कई भाषाएँ बोली जाती हैं और लोग अलग-अलग "
        "त्योहार बड़े उत्साह से मनाते हैं।",
        # te (Telugu)
        "భారతదేశం ఒక పెద్ద దేశం. ఇక్కడ చాలా భాషలు మాట్లాడతారు మరియు ప్రజలు వివిధ "
        "పండుగలను ఎంతో ఉత్సాహంగా జరుపుకుంటారు.",
        # cm (Hinglish, romanised)
        "Yaar kal main college gaya tha but professor nahi aaye. Phir humne "
        "canteen me chai pi aur thodi der baithe rahe.",
    ]
    buckets = ["en", "en", "hi", "te", "cm"]

    extractor = StylometricExtractor()
    matrix = extractor.fit_transform(samples, languages=buckets)
    names = extractor.feature_names()

    print(f"feature matrix shape: {matrix.shape}")
    print(f"n_features: {len(names)}")
    print("feature names:")
    for i, name in enumerate(names):
        print(f"  [{i:2d}] {name}")

    scores = extractor.score(samples)
    print("\nprovisional machine-likelihood scores (NOT calibrated):")
    for bucket, s in zip(buckets, scores):
        print(f"  {bucket}: {s:.3f}")
