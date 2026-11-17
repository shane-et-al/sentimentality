"""Compare out-of-the-box sentiment from VADER, BERT and SiEBERT, line by line, on a text file.

VADER: rule/lexicon-based, English-only. Reports a compound score in [-1, 1];
  the conventional cutoffs are >= 0.05 positive, <= -0.05 negative.
VADER-C: VADER's rules with the word ratings from VADER-C_lexicon.txt
  instead of VADER's built-in lexicon.
BERT: nlptown/bert-base-multilingual-uncased-sentiment, predicts 1-5 stars.
  To compare against VADER, the star probabilities are collapsed into an
  expected star rating and mapped linearly onto [-1, 1] (1★ -> -1, 3★ -> 0, 5★ -> +1).
SiEBERT: siebert/sentiment-roberta-large-english, RoBERTa-large fine-tuned on 15
  English datasets of varied text; predicts positive or negative. Its score is
  P(positive) - P(negative), in [-1, 1].

Both transformer models accept at most 512 tokens, so longer lines are cut off at
that point and flagged in the CSV; VADER always reads the whole line. Blank lines
are skipped.

Usage:
    python compare_sentiment.py                 # run on pg5200.txt
    python compare_sentiment.py other.txt       # run on another file
    python compare_sentiment.py --csv out.csv   # choose where the CSV goes
    python compare_sentiment.py --lexicon x.txt # choose the VADER-C lexicon
"""

import argparse
import csv
import os
from collections import Counter
from itertools import combinations
from statistics import correlation, mean

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
SIEBERT_MODEL = "siebert/sentiment-roberta-large-english"

# (display name, CSV prefix) for each tool; each has <prefix>_score and <prefix>_label columns
TOOLS = (("VADER", "vader"), ("VADER-C", "vader_claude"), ("BERT", "bert"), ("SiEBERT", "siebert"))


def label(score, threshold=0.05):
    if score >= threshold:
        return "pos"
    if score <= -threshold:
        return "neg"
    return "neu"


def classify(model, texts):
    """Yield each text's {label: probability} and whether it exceeded the model's token limit."""
    classifier = pipeline("sentiment-analysis", model=model, top_k=None)
    max_len = classifier.tokenizer.model_max_length
    for text, dist in zip(texts, classifier(list(texts), truncation=True)):
        truncated = len(classifier.tokenizer(text, verbose=False)["input_ids"]) > max_len
        yield {d["label"]: d["score"] for d in dist}, truncated


def bert_scores(texts):
    for dist, truncated in classify(BERT_MODEL, texts):
        probs = {int(name.split()[0]): p for name, p in dist.items()}
        top = max(probs, key=probs.get)
        expected = sum(stars * p for stars, p in probs.items())
        yield probs, top, (expected - 3) / 2, truncated


def siebert_scores(texts):
    for dist, truncated in classify(SIEBERT_MODEL, texts):
        yield dist["POSITIVE"], dist["POSITIVE"] - dist["NEGATIVE"], truncated


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default="pg5200.txt", help="text file, one passage per line")
    parser.add_argument("--csv", default="sentiment_comparison.csv", help="CSV output path")
    parser.add_argument("--lexicon", default="VADER-C_lexicon.txt", help="word ratings for VADER-C")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        numbered = [(n, line.strip()) for n, line in enumerate(f, 1) if line.strip()]
    line_nums, texts = zip(*numbered)
    vader = SentimentIntensityAnalyzer()
    # VADER joins lexicon_file onto its own package directory, so it must be absolute
    vader_claude = SentimentIntensityAnalyzer(lexicon_file=os.path.abspath(args.lexicon))
    rows = []

    print(f"{'line':>4}  {'VADER':>13}   {'VADER-C':>13}   {'BERT':>21}   {'SiEBERT':>13}   agree  text")
    results = zip(line_nums, texts, bert_scores(texts), siebert_scores(texts))
    for line, text, (probs, stars, bert_norm, bert_cut), (sie_pos, sie_norm, sie_cut) in results:
        vs = vader.polarity_scores(text)
        compound = vs["compound"]
        vc_score = vader_claude.polarity_scores(text)["compound"]
        v, vc = label(compound), label(vc_score)
        b, s = label(bert_norm, threshold=0.2), label(sie_norm, threshold=0.5)
        snippet = text if len(text) <= 60 else text[:57] + "..."
        print(
            f"{line:>4}  {compound:+.3f} ({v})   {vc_score:+.3f} ({vc})   {stars}★ {probs[stars]:4.0%} {bert_norm:+.3f} ({b})"
            f"   {sie_norm:+.3f} ({s})   {'all' if v == vc == b == s else '':>5}  {snippet}"
        )
        rows.append({
            "line": line,
            "vader_score": compound,
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "vader_pos": vs["pos"],
            "vader_label": v,
            "vader_claude_score": vc_score,
            "vader_claude_label": vc,
            "bert_stars": stars,
            **{f"bert_p{n}": round(probs[n], 4) for n in range(1, 6)},
            "bert_score": round(bert_norm, 4),
            "bert_label": b,
            "siebert_p_pos": round(sie_pos, 4),
            "siebert_score": round(sie_norm, 4),
            "siebert_label": s,
            "truncated": bert_cut or sie_cut,
        })
        row = rows[-1]
        for (_, a), (_, b_) in combinations(TOOLS, 2):
            row[f"agree_{a}_{b_}"] = row[f"{a}_label"] == row[f"{b_}_label"]
        row["text"] = text

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")
    summarize(rows)


def summarize(rows):
    labels = ("pos", "neu", "neg")

    print(f"\nSummary of {len(rows)} lines")
    print(f"  {'':8}{'pos':>5}{'neu':>5}{'neg':>5}   mean score")
    for name, key in TOOLS:
        counts = Counter(r[f"{key}_label"] for r in rows)
        print(f"  {name:8}" + "".join(f"{counts[l]:>5}" for l in labels)
              + f"   {mean(r[f'{key}_score'] for r in rows):+.3f}")

    print(f"\n  {'':19}{'agree':>11}{'opposite':>10}{'one neutral':>13}{'rank corr':>11}")
    for (a_name, a), (b_name, b) in combinations(TOOLS, 2):
        pairs = Counter((r[f"{a}_label"], r[f"{b}_label"]) for r in rows)
        agree = sum(pairs[l, l] for l in labels)
        opposite = pairs["pos", "neg"] + pairs["neg", "pos"]
        corr = (f"{correlation([r[f'{a}_score'] for r in rows], [r[f'{b}_score'] for r in rows], method='ranked'):+.2f}"
                if len(rows) > 1 else "")
        print(f"  {a_name + ' / ' + b_name:19}{agree:>5} ({agree / len(rows):3.0%}){opposite:>10}"
              f"{len(rows) - agree - opposite:>13}{corr:>11}")

    all_agree = sum(len({r[f"{key}_label"] for _, key in TOOLS}) == 1 for r in rows)
    print(f"\n  All agree:        {all_agree} of {len(rows)} ({all_agree / len(rows):.0%})")
    truncated = sum(r["truncated"] for r in rows)
    if truncated:
        print(f"  Truncated:        {truncated}")


if __name__ == "__main__":
    main()
