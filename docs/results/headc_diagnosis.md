# Head C shortcut diagnosis

Pre-fusion check (Review-2 Day 4): is Head C's 1.000 AUROC (en/hi/te) a real semantic signal or a shortcut on length / domain / a trivial embedding direction? All numbers are test-split, per bucket.


## en

- Head C (full MuRIL embedding, per-bucket logistic): **1.000**
- length_words alone: 0.565
- length_tokens alone: 0.566
- domain one-hot alone: 0.532 (5 domains: news_gov_mixed, qa_reddit_eli5, qa_wikiqa, wiki_cs_ai, wiki_general)
- embedding, top-10 PCA components: 0.994
- embedding L2 norm alone: 0.720
- TF-IDF unigram+bigram bag-of-words (no embeddings): **0.991**
- corr(PC1..3, length_words) = [-0.069, 0.03, -0.193]; corr(PC1..3, label) = [0.799, -0.223, 0.107]
- corr(embedding norm, length_words) = 0.308; corr(embedding norm, label) = 0.329
- **verdict: SHORTCUT (lexical fingerprint of the single seen generator)**

## hi

- Head C (full MuRIL embedding, per-bucket logistic): **1.000**
- length_words alone: 0.502
- length_tokens alone: 0.540
- domain one-hot alone: 0.518 (2 domains: news_web, wiki_general)
- embedding, top-10 PCA components: 0.998
- embedding L2 norm alone: 0.780
- TF-IDF unigram+bigram bag-of-words (no embeddings): **0.986**
- corr(PC1..3, length_words) = [0.22, 0.255, 0.178]; corr(PC1..3, label) = [0.746, -0.485, -0.21]
- corr(embedding norm, length_words) = 0.169; corr(embedding norm, label) = -0.428
- **verdict: SHORTCUT (lexical fingerprint of the single seen generator)**

## te

- Head C (full MuRIL embedding, per-bucket logistic): **1.000**
- length_words alone: 0.521
- length_tokens alone: 0.509
- domain one-hot alone: 0.499 (2 domains: news_web, wiki_general)
- embedding, top-10 PCA components: 1.000
- embedding L2 norm alone: 0.917
- TF-IDF unigram+bigram bag-of-words (no embeddings): **0.997**
- corr(PC1..3, length_words) = [0.03, 0.268, -0.109]; corr(PC1..3, label) = [-0.899, -0.174, 0.124]
- corr(embedding norm, length_words) = 0.186; corr(embedding norm, label) = -0.707
- **verdict: SHORTCUT (length/domain)**

## cm

- Head C (full MuRIL embedding, per-bucket logistic): **0.927**
- length_words alone: 0.500
- length_tokens alone: 0.567
- domain one-hot alone: 0.600 (3 domains: chat_dialogue, hinglish_generation, social_comments)
- embedding, top-10 PCA components: 0.909
- embedding L2 norm alone: 0.504
- TF-IDF unigram+bigram bag-of-words (no embeddings): **0.897**
- corr(PC1..3, length_words) = [-0.077, -0.032, 0.178]; corr(PC1..3, label) = [0.045, -0.086, 0.059]
- corr(embedding norm, length_words) = 0.131; corr(embedding norm, label) = -0.020
- **verdict: SHORTCUT (lexical fingerprint of the single seen generator)**
