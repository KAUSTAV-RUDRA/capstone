# Corpus card — IndicStudentMGT

**Corpus v2 is frozen (2026-09-29): human + qwen7b, gemma, mistral (seen) + llama (held-out).**
§0 describes the frozen corpus (`data/processed/corpus.jsonl`,
`data/processed/splits.json`); §1–§7 describe the human source text it was built
from (`data/raw/human/`). v1 (qwen7b only, the Review-2 corpus) is archived, not
deleted: see the hash table.

---

## 0. Frozen corpus v2

Built once by

```
python -m src.data.freeze_splits --config configs/data.yaml --allow-small-buckets
```

after v1 was moved to `data/processed/archive/v1/` (non-negotiable #6: each version is
written once and the script refuses to overwrite it). `data/` is gitignored, so the
committed evidence is the hashes:

| file | rows | sha256 |
|---|---|---|
| **v2** `data/processed/splits.json` | 12,148 ids | `b550d42a89f1a189a761e4e68c65a7f40822afe604d9bc0838f5becfc04bee1a` |
| **v2** `data/processed/corpus.jsonl` | 12,148 | `2cee2f8dd40cb07c6295ef8867c68d5858f7db62cca2ef54fd63191c75a4099b` |
| v1 (archived) `data/processed/archive/v1/splits.json` | 8,896 ids | `1a71fa2f0ddcd404c10cb8514680fb8e98240a985e2001a1766e78d502f17471` |
| v1 (archived) `data/processed/archive/v1/corpus.jsonl` | 8,896 | `9b0f10e988debbff78dbec39eac268aa9124b49a2ceeb1f2bca0eed9ceb657a0` |

The v1 hashes are unchanged from the 2026-09-24 card; they were re-checked after the
move. Every Review-2 number (`results/scores.parquet`, fitted heads) was computed on v1
and reproduces only against the archived files. **v2 is a different split, not an
extension of v1:** the calibration draw, prompt groups and length matching were re-run
over all generators, so ids move between splits and v1/v2 numbers are not comparable
row for row.

Rows carry the 11 schema fields (`id | text | label | language | code_mix_ratio |
generator | domain | length_tokens | attack_type | writer_L1_band | split`) then
`length_words | source | prompt_id`. `text` is the **cleaned** text; `length_tokens`
is counted with the Head B scorer's tokenizer (ai-forever/mGPT); `code_mix_ratio` is
recomputed from the cleaned text for both classes.

**Generators.** Seen: qwen7b (en, hi, te, cm), gemma (en, hi, te, cm), mistral (en, hi).
Held-out, test only: llama (hi). mistral does not generate te or cm, phi is dropped,
and llama does not generate cm or te (decisions.md 2026-09-24 to 2026-09-29). **T3 is
computable on hi only**; te and cm have no held-out generator.

### Pipeline

1. **Clean** (`src/data/clean_artifacts.py`), identically for human and machine:
   strip trailing assistant chatter, leading preambles ("Certainly!", "Here is a short
   passage of about 150 words…:"), markdown emphasis/headers, and `clean_informal` for
   cm; then drop rows with an emoji, instruction echo / meta response / chat-template
   leak, truncation (machine only), or a script the bucket does not use.
2. **Calibration draw**: 1,000 human rows per bucket, calibration-eligible sources
   only (cm: cmu_hinglish_dog + HinGE), stratified by source × length bin.
3. **Length-match** train/test: 5 quantile bins of train/test human word counts.
   en/hi/te trim **human** to the machine distribution (human is the surplus); cm trims
   **machine** to the human distribution.
4. **Split** by prompt group: a human passage and its machine continuation always
   share a split, and every seen generator is halved train/test within one row. A prompt
   that has a llama row is forced to `test`, together with its human passage and any
   seen-generator row, so no pair straddles train and test (0 prompts do).

### Cleaning — % dropped per bucket × source (v2)

| bucket | source | rows | dropped | % | emoji | echo | truncated | stray script |
|---|---|---|---|---|---|---|---|---|
| en | hc3 | 605 | 3 | 0.5 | 0 | 0 | – | 3 |
| en | samanantar_en | 1100 | 0 | 0.0 | 0 | 0 | – | 0 |
| en | wikipedia | 495 | 4 | 0.8 | 2 | 0 | – | 2 |
| en | **qwen7b** | 730 | 10 | 1.4 | 0 | 3 | 0 | 7 |
| en | **gemma** | 712 | 8 | 1.1 | 7 | 1 | 0 | 0 |
| en | **mistral** | 758 | 1 | 0.1 | 0 | 0 | 1 | 0 |
| hi | indiccorp_v2 | 1650 | 3 | 0.2 | 2 | 0 | – | 1 |
| hi | wikipedia | 550 | 30 | 5.5 | 0 | 0 | – | 30 |
| hi | **qwen7b** | 500 | 34 | 6.8 | 1 | 1 | 15 | 18 |
| hi | **gemma** | 250 | 1 | 0.4 | 0 | 0 | 0 | 1 |
| hi | **mistral** | 250 | 79 | 31.6 | 0 | 1 | 7 | 75 |
| hi | **llama** | 500 | 21 | 4.2 | 0 | 0 | 11 | 10 |
| te | indiccorp_v2 | 1650 | 4 | 0.2 | 2 | 0 | – | 2 |
| te | wikipedia | 550 | 39 | 7.1 | 0 | 0 | – | 39 |
| te | **qwen7b** | 400 | 72 | 18.0 | 0 | 2 | 3 | 69 |
| te | **gemma** | 200 | 79 | 39.5 | 2 | 1 | 0 | 78 |
| cm | cmu_hinglish_dog | 917 | 4 | 0.4 | 0 | 3 | – | 1 |
| cm | comi_lingua | 1175 | 4 | 0.3 | 0 | 0 | – | 4 |
| cm | hinge | 108 | 0 | 0.0 | 0 | 0 | – | 0 |
| cm | **qwen7b** | 701 | 52 | 7.4 | 3 | 25 | 13 | 15 |
| cm | **gemma** | 790 | 7 | 0.9 | 4 | 1 | 0 | 3 |

A row can fail more than one rule. 455 rows dropped in all, copied to
`data/raw/discarded/part13_clean_v2.jsonl`. The two large machine drop rates are both
stray script in an Indic bucket: **mistral hi 31.6 %** (75 of 250) and **gemma te
39.5 %** (78 of 200), on top of qwen7b te's 18.0 % (mostly Devanagari inside Telugu).
The surviving rows are what the model wrote in the right script, so the bucket's machine
text is biased towards passages the generator handled cleanly; report it with those
rates. Indic Wikipedia's human drops are names in other Indian scripts; symmetric
application costs those sources 5–7 % but stays under the 2 %-per-bucket human guard.

### Final counts — bucket × generator × split

| bucket | generator | cal | train | test | total |
|---|---|---|---|---|---|
| en | human | 1000 | 403 | 402 | 1805 |
| en | qwen7b | – | 359 | 361 | 720 |
| en | gemma | – | 352 | 352 | 704 |
| en | mistral | – | 379 | 378 | 757 |
| hi | human | 1000 | 452 | 449 | 1901 |
| hi | qwen7b | – | 233 | 233 | 466 |
| hi | gemma | – | 124 | 125 | 249 |
| hi | mistral | – | 85 | 86 | 171 |
| hi | llama (held-out) | – | 0 | 479 | 479 |
| te | human | 1000 | 394 | 397 | 1791 |
| te | qwen7b | – | 164 | 164 | 328 |
| te | gemma | – | 60 | 61 | 121 |
| cm | human | 1000 | 595 | 597 | 2192 |
| cm | qwen7b | – | 97 | 98 | 195 |
| cm | gemma | – | 135 | 134 | 269 |

**Total 12,148 rows** (7,689 human, 4,459 machine). Machine per bucket: en 2,181,
hi 1,365, te 449, cm 464. te and cm are below the 500-row floor; frozen under
`--allow-small-buckets` (decisions.md 2026-09-29). Length matching excluded 1,988 rows
(1,020 human in en/hi/te, 968 machine in cm); their ids and prompt groups are in
`splits.json` under `excluded.length_match`. Calibration: 1,000 human per bucket, and cm
cal is cmu_hinglish_dog (894) + HinGE (106) only.

### Length after matching (median words / mGPT tokens)

| bucket | cal human | test human | test machine |
|---|---|---|---|
| en | 208 / 283 | 200 / 273 | 206 / 290 |
| hi | 158 / 563 | 171 / 599 | 169 / 604 |
| te | 154 / 979.5 | 151 / 953 | 151 / 956 |
| cm | 85 / 146 | 20 / 35 | 20 / 35 |

Matching is against the **pooled** machine distribution of a bucket, not per generator,
so a generator can sit off the human median even though the pool matches. Test median
words, human vs machine: **en** human 200 against qwen7b 181, gemma 163.5, **mistral
247**; hi 171 against qwen7b 168, gemma 164, mistral 164.5, llama 173; te 151 against
qwen7b 157, gemma 133; cm 20 against qwen7b 20, gemma 19. en/mistral is 1.24× human and
en/gemma 0.82×, so any per-generator T3/T4 row for en carries a length caveat.

### cm: the bound is calibrated on chat-register Hinglish

cm calibration can only draw from the two date-certain sources, which together barely
clear the 1,000 floor, so cal is chat turns (cmu_hinglish_dog, median 85 words) and cm's
human train/test is 98 % social comments (comi_lingua, median 20). **cm's conformal
bound is calibrated on chat-register Hinglish and holds for that register.** It is not
guaranteed on comment-register text, and cm's test FPR is reported as measured, not as
bounded. This is the project's own thesis showing up inside the corpus: a threshold is
only as good as the match between its calibration population and its deployment
population (decisions.md 2026-09-24, "Finding"). Every table row for cm carries this
note.

In en/hi/te the same effect is mild: trimming human train/test to the machine
lengths shifts the test humans' length distribution relative to cal (table above).

---

## Human source text (`data/raw/human/`)

The human half: 8,800 passages across four language buckets, built by
`src/data/build_human_corpus.py` from `configs/data.yaml`.

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
