# Corpus card — human text (`data/raw/human/`)

The human half of **IndicStudentMGT**: 8,800 passages across four language
buckets, built by `src/data/build_human_corpus.py` from `configs/data.yaml`.
Machine text, length matching and the frozen splits are added later (parts-plan
Parts 4–13); this card covers the human side only.

Regenerate any bucket with:

```
python -m src.data.build_human_corpus --bucket <en|hi|te|cm> --target 2200 --config configs/data.yaml
```

The builder is resumable and deterministic: ids are derived from the source
document, so re-running tops a bucket up rather than duplicating it, and a
completed bucket re-runs as a no-op.

Last built: 2026-09-13.

---

## 1. Summary

| bucket | rows | bands | mean words | median | min–max | script purity | duplicate texts |
|---|---|---|---|---|---|---|---|
| en | 2200 | general 1100, indian 1100 | 210.6 | 207 | 120–300 | 1.000 (min 0.97) | 0 |
| hi | 2200 | native 2200 | 175.6 | 159 | 120–300 | 0.991 (min 0.81) | 0 |
| te | 2200 | native 2200 | 170.3 | 155 | 120–300 | 0.991 (min 0.80) | 0 |
| cm | 2200 | native 2200 | 55.2 | 25 | 15–298 | 1.000 (min 0.90) | 0 |

**Total 8,800 human passages.** Every bucket clears the 1,000-passage
calibration floor (locked decision §5) with ≥1,200 left for train/test.

*Script purity* is the fraction of letters in the bucket's own script
(Devanagari for hi, Telugu for te, Latin for en and cm). It is a ratio rather
than a hard rule because Latin loanwords and acronyms are normal in Indian
writing.

---

## 2. Sources and licences

| bucket | source | domain | rows | % | licence | provenance and date |
|---|---|---|---|---|---|---|
| en | `ai4bharat/samanantar` (English side, hi config) | news_gov_mixed | 1100 | 50 % | CC-BY-NC-4.0 | Indian news / PIB / PMIndia, ≤ 2020 |
| en | `wikimedia/wikipedia` 20231101.en, page id ≤ 2,000,000 | wiki_general | 495 | 22 % | CC-BY-SA-4.0 | articles created ≤ 2005; text as of the 2023-11-01 dump |
| en | `Hello-SimpleAI/HC3` human answers (reddit_eli5) | qa_reddit_eli5 | 304 | 14 % | CC-BY-SA-4.0 | Reddit ELI5, 2011–2019 |
| en | `Hello-SimpleAI/HC3` human answers (wiki_csai) | wiki_cs_ai | 300 | 14 % | CC-BY-SA-4.0 | Wikipedia CS/AI intros, ≤ 2022 |
| en | `Hello-SimpleAI/HC3` human answers (open_qa) | qa_wikiqa | 1 | 0 % | CC-BY-SA-4.0 | WikiQA, 2015 |
| hi | `ai4bharat/IndicCorpV2`, split `hin_Deva` | news_web | 1650 | 75 % | **CC-0** | crawled Indian news/web, crawl ≤ 2022 |
| hi | `wikimedia/wikipedia` 20231101.hi, page id ≤ 100,000 | wiki_general | 550 | 25 % | CC-BY-SA-4.0 | early-created articles |
| te | `ai4bharat/IndicCorpV2`, split `tel_Telu` | news_web | 1650 | 75 % | **CC-0** | crawled Indian news/web, crawl ≤ 2022 |
| te | `wikimedia/wikipedia` 20231101.te, page id ≤ 100,000 | wiki_general | 550 | 25 % | CC-BY-SA-4.0 | early-created articles |
| cm | `LingoIITGN/COMI-LINGUA`, TN config, raw `Sentences` | social_comments | 1175 | 53 % | CC-BY-4.0 | social comments; **collection date not stated** |
| cm | `festvox/cmu_hinglish_dog` | chat_dialogue | 917 | 42 % | CC-BY-SA-3.0 + GFDL | two-person chat, 2018–2020 |
| cm | `LingoIITGN/HinGE` | hinglish_generation | 108 | 5 % | CC-BY-4.0 | human-written Hinglish, 2021 |

**Redistribution.** Samanantar is **non-commercial (CC-BY-NC-4.0)**, so the
assembled corpus cannot be redistributed commercially as-is. Everything else is
CC-0, CC-BY or CC-BY-SA. If a commercially-redistributable release is ever
needed, the `en`/`indian` band has to be rebuilt from a different source; that
is a one-line change in `configs/data.yaml`.

### Per-source length

| bucket | source | rows | mean words | median |
|---|---|---|---|---|
| en | hc3 (reddit_eli5) | 304 | 192.2 | 184 |
| en | hc3 (wiki_csai) | 300 | 187.2 | 178 |
| en | samanantar_en | 1100 | 218.4 | 218 |
| en | wikipedia | 495 | 218.9 | 220 |
| hi | indiccorp_v2 | 1650 | 163.9 | 150 |
| hi | wikipedia | 550 | 210.5 | 209 |
| te | indiccorp_v2 | 1650 | 161.4 | 148 |
| te | wikipedia | 550 | 196.9 | 189 |
| cm | cmu_hinglish_dog | 917 | 103.1 | 93 |
| cm | comi_lingua | 1175 | 20.8 | 20 |
| cm | hinge | 108 | 23.4 | 20 |

---

## 3. Writer bands

`writer_L1_band` drives the fairness table (T5).

| band | buckets | meaning |
|---|---|---|
| `general` | en (1100) | General English, the L1 proxy. |
| `indian` | en (1100) | Indian-authored English, the L2 proxy. |
| `native` | hi, te, cm (2200 each) | First-language writers. |

The `general` / `indian` split exists only for `en`, because the fairness claim
this project makes is about non-native **English** being misflagged. No public
source carries real L1 metadata, so authorship origin is the proxy.

---

## 4. Construction rules

- **One passage per source document.** Leading sentences up to a per-document
  target length drawn deterministically from the document id, so long sources do
  not all pile up on the cap.
- **Length 120–300 words**, except `cm`, which is **15–300** (see §6).
- **Near-duplicate removal** with MinHash LSH (`datasketch`): 128 permutations,
  word 5-gram shingles, Jaccard ≥ 0.8. The index is rebuilt from the file on
  every run, so it also catches duplicates across runs.
- **Script purity** ≥ 0.8 of letters for hi and te, ≥ 0.9 for cm.
- **Code-mixing gate** for `cm` only: ≥ 0.3 of word tokens must be Romanised
  Hindi function words. The measured value is stored per row as
  `code_mix_ratio` (min 0.30, median 0.39, p90 0.50, max 0.73, mean 0.403).
  It is 0.0 in every other bucket.

### Cleaning

Human text must not be separable from machine text by a surface cue, or every
AUROC in the paper is inflated by an artefact. Three cleaners run before a
passage is accepted, and anything still carrying an artefact is rejected:

| cleaner | applies to | fixes |
|---|---|---|
| `repair_holes` | wikipedia, hc3 | template holes: `Albedo (; ) is` → `Albedo is`, `Lincoln ( ; February 12` → `Lincoln (February 12` |
| `detokenise_ptb` | hc3 reddit_eli5, open_qa | PTB tokenisation: `word , it 's ( x )` → `word, it's (x)`; strips `URL_0` placeholders |
| `clean_informal` | all cm sources | chat spacing: `kaisa hai ?` → `kaisa hai?` |

Rejected outright: residual space-before-punctuation (Latin or danda), LaTeX
left by Wikipedia scrapes (`\displaystyle`), wiki markup, and U+FFFD.

> **Carry-over for `clean_artifacts.py` (Part 13):** machine `cm` text must get
> the same `clean_informal` normalisation, or the asymmetry becomes a shortcut
> feature in the opposite direction.

### Cleaning statistics

Documents read versus passages kept, from the build runs. Percentages are of
documents read.

| bucket | source | read | kept | rejected | main reasons |
|---|---|---|---|---|---|
| en | hc3 | 2,195 | 605 | 72 % | too short 58 %, artefact ~1 %, near-duplicate <1 % |
| en | wikipedia | 983 | 495 | 50 % | artefact 21 %, too short 5 % |
| en | samanantar_en | 1,600 | 1,100 | 31 % | already present (resumed run) |
| hi | indiccorp_v2 | 24,429 | 1,650 | 93.2 % | too short 92.1 %, artefact 1.0 %, wrong script 0.1 % |
| hi | wikipedia | 936 | 550 | 41.2 % | artefact 17.0 %, too short 24.0 % |
| te | indiccorp_v2 | 79,770 | 1,650 | 97.9 % | too short 97.4 %, artefact 0.5 % |
| te | wikipedia | 1,263 | 550 | 56.5 % | artefact 39.3 %, too short 16.8 % |
| cm | hinge | 1,976 | 108 | 94.5 % | too short 52.1 %, not code-mixed 42.1 % |
| cm | cmu_hinglish_dog | 1,093 | 917 | 16.1 % | not code-mixed 9.5 %, too short 4.8 % |
| cm | comi_lingua | 2,546 | 1,175 | 53.8 % | not code-mixed 27.7 %, too short 25.4 % |

The `en` figures aggregate two build runs (target 1,000 then 2,200), so "read"
there includes documents skipped as already present. Every other bucket was
built in a single run.

Two rates are worth noting. **Length dominates the Indic rejections** — only
~7 % of Hindi and ~2 % of Telugu IndicCorpV2 documents reach 120 words, which is
why `max_docs` is set far above the row target. **Indic Wikipedia carries far
more template holes than English** — 17 % of Hindi and 39 % of Telugu articles
read were rejected as artefacts, against 21 % for English.

---

## 5. Row schema

A superset of `src/data/schema.py`. `split` is assigned later by
`freeze_splits.py`; `length_tokens` is filled at scoring time.

```
id | text | label=0 | language | code_mix_ratio | generator=null | domain |
length_words | attack_type="clean" | writer_L1_band | split=null |
source | source_id | provenance
```

Ids are `<bucket>_<source>_<source_id>`, e.g. `hi_indiccorp_v2_hin_Deva:2`.

---

## 6. Known limitations

These are corpus properties that affect how results must be read. Each one
belongs in the paper's Limitations section.

1. **`cm` passages are much shorter than the rest** — mean 55.2 words and median
   25, against 170–211 for the other buckets. Romanised Hinglish occurs as
   comments and chat turns; a 120-word floor would have rejected essentially all
   of it. Because curvature scores depend strongly on length, a cross-bucket
   comparison of `cm` against `en` confounds language with length. Length
   matching in Part 13 equalises machine against human *within* a bucket, not
   across buckets, so any per-bucket table that includes `cm` needs either a
   length-controlled sub-analysis or an explicit caveat.

2. **`cm` is internally bimodal** — chat transcripts average 103 words while
   social comments and HinGE sentences average ~21. The bucket is not one
   population.

3. **Samanantar is shuffled sentences, not documents.** Checked at six offsets:
   every region is an unordered mix of unrelated sentences. The `en`/`indian`
   passages are runs of consecutive *filtered* sentences — authentically
   Indian-authored at sentence level, but topically incoherent at passage level.
   Head A's discourse features (burstiness, type-token ratio) will see unusually
   diverse text in exactly the band the fairness claim rests on.

4. **COMI-LINGUA's collection date is not stated**, so machine-written text
   cannot be ruled out in 53 % of the `cm` bucket. The other two `cm` sources
   (2018–2021) and every en/hi/te source predate ChatGPT. This is the weakest
   provenance in the corpus.

5. **The `general` / `indian` split is an authorship proxy, not measured L1.**
   No public source carries writer L1 metadata.

6. **The Hindi word list detects Hindi-matrix code-mixing, not English-matrix.**
   It is built from function words, which carry the grammar. HinGE instead
   substitutes Hindi *content* words into English sentences ("a part of vikas of
   the adhosanrachna"), which scores low: 42 % of HinGE rows were rejected as
   not code-mixed, and its median ratio is 0.18 against 0.39 for the chat data.
   The `cm` bucket therefore under-represents English-matrix Hinglish.

7. **Domain is not balanced across buckets.** `en` is a four-domain mix, `hi`
   and `te` are 75 % news / 25 % encyclopedic, `cm` is chat and social comments.
   Some of any per-bucket difference is a domain difference.

8. **Wikipedia "low page id" is a proxy for creation date, not a guarantee.**
   Article *text* is as of the 2023-11-01 dump, so an old article edited recently
   carries recent text.

---

## 7. Ethics and intended use

- All sources are public and openly licensed; no personal data was collected for
  this project and no scraping was performed.
- No real student submissions are included. Ethics approval for those had not
  cleared, and proxy corpora are the primary path.
- The corpus exists to evaluate a **decision-support** detector that abstains
  when unsure. It must not be used to build a system that issues automatic
  accusations (non-negotiable #8).
- Text is reproduced from its sources under the licences above; attribution
  belongs with the original datasets, cited in the paper.
