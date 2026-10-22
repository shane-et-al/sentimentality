# sentimentality

Compares two off-the-shelf sentiment analysis tools, line by line, on a text file. By default the file is `pg5200.txt`, David Wyllie's translation of Franz Kafka's *The Metamorphosis* edited to contain one paragraph per line.

- **VADER**: a rule- and word-list-based scorer. English only.
- **BERT**: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), a model trained on product reviews that predicts a 1–5 star rating. Works in English, Spanish, French, German, Dutch and Italian.

## Usage

```sh
uv run python compare_sentiment.py                # compare each line of pg5200.txt
uv run python compare_sentiment.py other.txt      # compare each line of another file
uv run python compare_sentiment.py --csv out.csv  # write the CSV somewhere else
```

Blank lines are skipped. The script prints a summary table and writes one row per line to `sentiment_comparison.csv`. Each run overwrites the file.

BERT can read at most 512 tokens (roughly 350–400 English words), so for longer lines it only sees the beginning. These lines are marked in the `bert_truncated` column. VADER always reads the whole line.

## CSV columns

| Column | Range | Meaning |
|---|---|---|
| `line` | 1 and up | Line number in the input file. |
| `text` | | The line's text, in full. |
| `vader_compound` | −1 to +1 | VADER's overall score. It adds up the sentiment of individual words and squashes the total into −1…+1, so longer texts with many emotional words tend toward the ends. |
| `vader_neg` | 0 to 1 | Share of the text VADER reads as negative. |
| `vader_neu` | 0 to 1 | Share of the text VADER reads as neutral. Text in a language VADER doesn't know scores 1.0 here. |
| `vader_pos` | 0 to 1 | Share of the text VADER reads as positive. `vader_neg`, `vader_neu` and `vader_pos` add up to about 1. |
| `vader_label` | `pos` / `neu` / `neg` | `pos` if `vader_compound` ≥ 0.05, `neg` if ≤ −0.05, otherwise `neu`. These are VADER's standard cutoffs. |
| `bert_stars` | 1 to 5 | BERT's single most likely star rating. |
| `bert_p1` … `bert_p5` | 0 to 1 | How likely BERT thinks each star rating is. The five add up to 1. |
| `bert_score` | −1 to +1 | BERT's output on the same scale as `vader_compound`: the probability-weighted average star rating, mapped so that 1★ = −1, 3★ = 0 and 5★ = +1. |
| `bert_label` | `pos` / `neu` / `neg` | `pos` if `bert_score` ≥ 0.2, `neg` if ≤ −0.2, otherwise `neu`. The ±0.2 cutoff was chosen for this project, not taken from the model. |
| `bert_truncated` | `True` / `False` | Whether the line was too long for BERT, so its BERT scores are based only on the beginning of the line. |
| `agree` | `True` / `False` | Whether `vader_label` and `bert_label` match. |

### Comparing the scores

`vader_compound` and `bert_score` share a range but measure different things, so equal numbers don't mean equal strength of sentiment. Compare their sign and their ordering across texts, not their exact values. `agree` also depends on the label cutoffs: moving BERT's ±0.2 line changes whether lukewarm texts count as agreeing.
