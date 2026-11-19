"""Compare out-of-the-box sentiment from VADER, BERT, SiEBERT and RoBERTa, line by line, on a text file.

Each line of the input file is treated as one passage. Every passage is scored
by five tools, the per-line comparison is printed, and the results are written
to a JSON file, one row per line, for index.html to display:

    VADER: rule/lexicon-based, English-only. Reports a compound score in
        [-1, 1]; the conventional cutoffs are >= 0.05 positive, <= -0.05
        negative.
    VADER-C: VADER's rules with the word ratings from VADER-C_lexicon.txt
        instead of VADER's built-in lexicon.
    BERT: nlptown/bert-base-multilingual-uncased-sentiment, predicts 1-5 stars.
        To compare against VADER, the star probabilities are collapsed into an
        expected star rating and mapped linearly onto [-1, 1] (1★ -> -1,
        3★ -> 0, 5★ -> +1).
    SiEBERT: siebert/sentiment-roberta-large-english, RoBERTa-large fine-tuned
        on 15 English datasets of varied text; predicts positive or negative.
        Its score is P(positive) - P(negative), in [-1, 1].
    RoBERTa: cardiffnlp/twitter-roberta-base-sentiment-latest, RoBERTa-base
        trained on tweets; predicts negative, neutral or positive. Its score
        is P(positive) - P(negative), in [-1, 1], and its label is the class it
        rates most likely rather than a cutoff on the score.

All three transformer models accept at most 512 tokens, so longer lines are cut off at
that point and flagged in the JSON; VADER always reads the whole line. Blank lines
are skipped.

Examples:
    Compare each line of pg5200.txt (the default), writing
    sentiment_comparison.json::

        python compare_sentiment.py

    Compare another file, choosing where the JSON goes and which word ratings
    to use for VADER-C::

        python compare_sentiment.py other.txt --json out.json --lexicon x.txt
"""

import argparse
import json
import os
from collections import Counter
from itertools import combinations
from statistics import correlation, mean

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = "pg5200.txt"
BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
SIEBERT_MODEL = "siebert/sentiment-roberta-large-english"
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Longest input the transformer models accept; some tokenizers don't declare it
MAX_TOKENS = 512

# (display name, JSON key) for each tool; each row has <key>.score and <key>.label
TOOLS = (
    ("VADER", "vader"), ("VADER-C", "vader_claude"), ("BERT", "bert"), ("SiEBERT", "siebert"), ("RoBERTa", "roberta"),
)


def label(score, threshold=0.05):
    """Turn a score in [-1, 1] into a pos/neu/neg label.

    Args:
        score (float): Sentiment score in [-1, 1].
        threshold (float): Scores >= threshold label as "pos", scores
            <= -threshold as "neg", and anything in between as "neu".

    Returns:
        str: "pos", "neg" or "neu".
    """
    if score >= threshold:
        return "pos"
    if score <= -threshold:
        return "neg"
    return "neu"


def classify(model, texts):
    """Score texts with a Hugging Face sentiment-analysis pipeline.

    Args:
        model (str): Hugging Face model name.
        texts (Iterable[str]): Texts to score, in order.

    Yields:
        tuple[dict[str, float], bool]: For each text, its label -> probability
        distribution and whether the text exceeded the model's maximum token
        count, in which case the pipeline truncated it.
    """
    classifier = pipeline("sentiment-analysis", model=model, top_k=None)
    max_len = min(classifier.tokenizer.model_max_length, MAX_TOKENS)
    for text, dist in zip(texts, classifier(list(texts), truncation=True, max_length=max_len)):
        truncated = len(classifier.tokenizer(text, verbose=False)["input_ids"]) > max_len
        yield {d["label"]: d["score"] for d in dist}, truncated


def bert_scores(texts):
    """Score texts with the BERT review model, normalized onto [-1, 1].

    Args:
        texts (Iterable[str]): Texts to score, in order.

    Yields:
        tuple[dict[int, float], int, float, bool]: For each text, in order: the
        probability of each star rating (keys 1-5), the most likely star
        rating, the probability-weighted average star rating mapped linearly
        onto [-1, 1] (1★ -> -1, 3★ -> 0, 5★ -> +1), and whether the text was
        truncated.
    """
    for dist, truncated in classify(BERT_MODEL, texts):
        probs = {int(name.split()[0]): p for name, p in dist.items()}
        top = max(probs, key=probs.get)
        expected = sum(stars * p for stars, p in probs.items())
        yield probs, top, (expected - 3) / 2, truncated


def siebert_scores(texts):
    """Score texts with SiEBERT, normalized onto [-1, 1].

    Args:
        texts (Iterable[str]): Texts to score, in order.

    Yields:
        tuple[float, float, bool]: For each text, in order: P(positive),
        P(positive) - P(negative) in [-1, 1], and whether the text was
        truncated.
    """
    for dist, truncated in classify(SIEBERT_MODEL, texts):
        yield dist["POSITIVE"], dist["POSITIVE"] - dist["NEGATIVE"], truncated


def roberta_scores(texts):
    """Score texts with the Twitter RoBERTa model, normalized onto [-1, 1].

    Args:
        texts (Iterable[str]): Texts to score, in order.

    Yields:
        tuple[dict[str, float], float, str, bool]: For each text, in order: the
        probability of each class (keys "negative", "neutral", "positive"),
        P(positive) - P(negative) in [-1, 1], the most likely class as
        "pos"/"neu"/"neg", and whether the text was truncated.
    """
    for dist, truncated in classify(ROBERTA_MODEL, texts):
        top = max(dist, key=dist.get)
        yield dist, dist["positive"] - dist["negative"], top[:3], truncated


def main():
    """Compare all five tools on every non-blank line and write the JSON.

    Command-line arguments (see ``--help`` and the module docstring): an
    optional positional ``file`` names the text to score (one passage per
    line, default pg5200.txt), ``--json`` sets the JSON output path (default
    sentiment_comparison.json, overwritten on every run), and ``--lexicon``
    sets the word ratings used for VADER-C (default VADER-C_lexicon.txt).

    Prints a per-line comparison table, writes the JSON (see the README for
    its format), then prints a summary of label counts and pairwise agreement
    (see ``summarize``).

    Raises:
        FileNotFoundError: If the input file or the VADER-C lexicon does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=INPUT_FILE, help="text file, one passage per line")
    parser.add_argument("--json", default="sentiment_comparison.json", help="JSON output path")
    parser.add_argument("--lexicon", default="VADER-C_lexicon.txt", help="word ratings for VADER-C")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        numbered = [(n, line.strip()) for n, line in enumerate(f, 1) if line.strip()]
    line_nums, texts = zip(*numbered)
    vader = SentimentIntensityAnalyzer()
    # VADER joins lexicon_file onto its own package directory, so it must be absolute
    vader_claude = SentimentIntensityAnalyzer(lexicon_file=os.path.abspath(args.lexicon))
    rows = []

    print(f"{'line':>4}  {'VADER':>13}   {'VADER-C':>13}   {'BERT':>21}   {'SiEBERT':>13}   {'RoBERTa':>13}"
          f"   agree  text")
    results = zip(line_nums, texts, bert_scores(texts), siebert_scores(texts), roberta_scores(texts))
    for line, text, (probs, stars, bert_norm, bert_cut), (sie_pos, sie_norm, sie_cut), \
            (rob_probs, rob_norm, r, rob_cut) in results:
        vs = vader.polarity_scores(text)
        compound = vs["compound"]
        vc_score = vader_claude.polarity_scores(text)["compound"]
        v, vc = label(compound), label(vc_score)
        b, s = label(bert_norm, threshold=0.2), label(sie_norm, threshold=0.5)
        snippet = text if len(text) <= 60 else text[:57] + "..."
        print(
            f"{line:>4}  {compound:+.3f} ({v})   {vc_score:+.3f} ({vc})   {stars}★ {probs[stars]:4.0%} {bert_norm:+.3f} ({b})"
            f"   {sie_norm:+.3f} ({s})   {rob_norm:+.3f} ({r})   {'all' if v == vc == b == s == r else '':>5}  {snippet}"
        )
        rows.append({
            "line": line,
            "text": text,
            "truncated": bert_cut or sie_cut or rob_cut,
            "vader": {"score": compound, "label": v, "neg": vs["neg"], "neu": vs["neu"], "pos": vs["pos"]},
            "vader_claude": {"score": vc_score, "label": vc},
            "bert": {"score": round(bert_norm, 4), "label": b, "stars": stars,
                     "probs": [round(probs[n], 4) for n in range(1, 6)]},
            "siebert": {"score": round(sie_norm, 4), "label": s, "p_pos": round(sie_pos, 4)},
            "roberta": {"score": round(rob_norm, 4), "label": r,
                        "probs": {name[:3]: round(rob_probs[name], 4) for name in ("negative", "neutral", "positive")}},
        })

    # One row per line of the file, so the output stays readable and diffs cleanly
    with open(args.json, "w", encoding="utf-8") as f:
        f.write(f'{{"source": {json.dumps(os.path.basename(args.file))}, '
                f'"lexicon": {json.dumps(os.path.basename(args.lexicon))}, "rows": [\n')
        f.write(",\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
        f.write("\n]}\n")
    print(f"\nWrote {len(rows)} rows to {args.json}")
    summarize(rows)


def summarize(rows):
    """Print summary statistics over the rows collected by ``main``.

    Prints, per tool: label counts and mean score. Then, per pair of tools:
    agreement (same label), disagreement (one said pos, the other neg), the
    remainder (at least one of the pair called the line neutral), and Spearman
    rank correlation of the scores. Finally, how often all five tools agreed
    and how many lines were truncated.

    Args:
        rows (list[dict]): The rows collected in ``main``: one dict per line
            with a "score" and "label" for each tool under its key ("vader",
            "vader_claude", "bert", "siebert", "roberta"), a boolean
            "truncated" key, and the line's "text".
    """
    labels = ("pos", "neu", "neg")

    print(f"\nSummary of {len(rows)} lines")
    print(f"  {'':8}{'pos':>5}{'neu':>5}{'neg':>5}   mean score")
    for name, key in TOOLS:
        counts = Counter(r[key]["label"] for r in rows)
        print(f"  {name:8}" + "".join(f"{counts[l]:>5}" for l in labels)
              + f"   {mean(r[key]['score'] for r in rows):+.3f}")

    print(f"\n  {'':19}{'agree':>11}{'opposite':>10}{'one neutral':>13}{'rank corr':>11}")
    for (a_name, a), (b_name, b) in combinations(TOOLS, 2):
        pairs = Counter((r[a]["label"], r[b]["label"]) for r in rows)
        agree = sum(pairs[l, l] for l in labels)
        opposite = pairs["pos", "neg"] + pairs["neg", "pos"]
        corr = (f"{correlation([r[a]['score'] for r in rows], [r[b]['score'] for r in rows], method='ranked'):+.2f}"
                if len(rows) > 1 else "")
        print(f"  {a_name + ' / ' + b_name:19}{agree:>5} ({agree / len(rows):3.0%}){opposite:>10}"
              f"{len(rows) - agree - opposite:>13}{corr:>11}")

    all_agree = sum(len({r[key]["label"] for _, key in TOOLS}) == 1 for r in rows)
    print(f"\n  All agree:        {all_agree} of {len(rows)} ({all_agree / len(rows):.0%})")
    truncated = sum(r["truncated"] for r in rows)
    if truncated:
        print(f"  Truncated:        {truncated}")


if __name__ == "__main__":
    main()
