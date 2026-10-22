"""Compare out-of-the-box sentiment from VADER and BERT, line by line, on a text file.

VADER: rule/lexicon-based, English-only. Reports a compound score in [-1, 1];
  the conventional cutoffs are >= 0.05 positive, <= -0.05 negative.
BERT: nlptown/bert-base-multilingual-uncased-sentiment, predicts 1-5 stars.
  To compare against VADER, the star probabilities are collapsed into an
  expected star rating and mapped linearly onto [-1, 1] (1★ -> -1, 3★ -> 0, 5★ -> +1).

BERT accepts at most 512 tokens, so longer lines are cut off at that point and
flagged in the CSV; VADER always reads the whole line. Blank lines are skipped.

Usage:
    python compare_sentiment.py                 # run on pg5200.txt
    python compare_sentiment.py other.txt       # run on another file
    python compare_sentiment.py --csv out.csv   # choose where the CSV goes
"""

import argparse
import csv

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"


def label(score, threshold=0.05):
    if score >= threshold:
        return "pos"
    if score <= -threshold:
        return "neg"
    return "neu"


def bert_scores(texts):
    classifier = pipeline("sentiment-analysis", model=MODEL, top_k=None)
    max_len = classifier.tokenizer.model_max_length
    for text, dist in zip(texts, classifier(list(texts), truncation=True)):
        probs = {int(d["label"].split()[0]): d["score"] for d in dist}
        top = max(probs, key=probs.get)
        expected = sum(stars * p for stars, p in probs.items())
        truncated = len(classifier.tokenizer(text, verbose=False)["input_ids"]) > max_len
        yield probs, top, (expected - 3) / 2, truncated


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default="pg5200.txt", help="text file, one passage per line")
    parser.add_argument("--csv", default="sentiment_comparison.csv", help="CSV output path")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        numbered = [(n, line.strip()) for n, line in enumerate(f, 1) if line.strip()]
    line_nums, texts = zip(*numbered)
    vader = SentimentIntensityAnalyzer()
    rows = []

    print(f"{'line':>4}  {'VADER':>13}   {'BERT':>21}   agree  text")
    for line, text, (probs, stars, bert_norm, truncated) in zip(line_nums, texts, bert_scores(texts)):
        vs = vader.polarity_scores(text)
        compound = vs["compound"]
        v, b = label(compound), label(bert_norm, threshold=0.2)
        snippet = text if len(text) <= 60 else text[:57] + "..."
        print(
            f"{line:>4}  {compound:+.3f} ({v})   {stars}★ {probs[stars]:4.0%} {bert_norm:+.3f} ({b})"
            f"   {'yes' if v == b else 'NO':>5}  {snippet}"
        )
        rows.append({
            "line": line,
            "vader_compound": compound,
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "vader_pos": vs["pos"],
            "vader_label": v,
            "bert_stars": stars,
            **{f"bert_p{n}": round(probs[n], 4) for n in range(1, 6)},
            "bert_score": round(bert_norm, 4),
            "bert_label": b,
            "bert_truncated": truncated,
            "agree": v == b,
            "text": text,

        })

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
