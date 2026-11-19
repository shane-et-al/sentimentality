# sentimentality

Sentiment analysis tool comparison and visualization. 

By default, the file is `pg5200.txt`, David Wyllie's translation of Franz Kafka's *The Metamorphosis* from Project Gutenberg, edited to contain one sentence per line/chunk to avoid the transformer models' token limits. To shake things up, `pg5200-3.txt`, a three-sentence per line text file is included but not used that gives each chunk more context.

- **VADER**: a rule- and word-list-based scorer. English only.
- **VADER-C**: VADER with its word ratings replaced by those in `VADER-C_lexicon.txt`. It has the same 7,506 words and the same rules, with different ratings for most words.
- **BERT**: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), a model trained on product reviews that predicts a 1–5 star rating. Works in English, Spanish, French, German, Dutch and Italian.
- **SiEBERT**: [`siebert/sentiment-roberta-large-english`](https://huggingface.co/siebert/sentiment-roberta-large-english), a larger model trained on 15 English datasets covering many kinds of text, not just reviews. It only predicts positive or negative. English only.
- **RoBERTa**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a model trained on tweets that predicts negative, neutral or positive. Unlike SiEBERT it has a real neutral class. English only.

## Usage

```sh
uv run python compare_sentiment.py                # compare each line of pg5200.txt
uv run python compare_sentiment.py other.txt      # compare each line of another file
uv run python compare_sentiment.py --csv out.csv  # write the CSV somewhere else
uv run python compare_sentiment.py --lexicon x.txt  # use another lexicon for VADER-C
```

Blank lines are skipped. The script prints a summary table and writes one row per line to `sentiment_comparison.csv`. Each run overwrites the file.

BERT, SiEBERT and RoBERTa can read at most 512 tokens (roughly 350–400 English words), so for longer lines they only see the beginning. These lines are marked in the `truncated` column. VADER always reads the whole line.

## HTML visualization

```sh
uv run python make_html.py                 # sentiment_comparison.csv -> sentiment_comparison.html
uv run python make_html.py other.csv       # visualize another CSV
uv run python make_html.py --source pg5200.txt --open   # label the source in the header and open it
```

Writes a single self-contained HTML file (no internet or libraries needed) with:

- A tool picker in the header: click a tool to hide or show it. Every card,
  the chart, the chunk list and its filter options are recalculated for the
  tools shown. At least two stay selected, and the choice is remembered in
  your browser.
- Summary cards: how often the tools agree, per-tool label distributions,
  pairwise agreement/opposition/rank-correlation, and the most common
  whole-chunk label patterns. Click any of them to filter the chunk list.
- A score timeline: one panel per tool over all chunks, with each tool's
  neutral zone shaded. Hover for values, click a point to inspect the chunk.
- A chunk-by-chunk heat strip: each chunk is a row, the tools are columns,
  cells are colored by label (green pos / gray neu / red neg, darker = stronger).
  Rows where tools disagree are highlighted; agreeing rows fade back; the left
  gutter shows how many distinct labels each chunk got. Filter by disagreement
  type or pair, search the text, and sort by score spread to surface the
  biggest differences first.
- A detail panel (click any chunk) with the full text and exact scores, BERT's
  star probabilities, VADER's word shares, SiEBERT's certainty and RoBERTa's
  class probabilities.

## CSV columns

| Column | Range | Meaning |
|---|---|---|
| `line` | 1 and up | Line number in the input file. |
| `vader_score` | −1 to +1 | VADER's overall score. It adds up the sentiment of individual words and squashes the total into −1…+1, so longer texts with many emotional words tend toward the ends. |
| `vader_neg` | 0 to 1 | Share of the text VADER reads as negative. |
| `vader_neu` | 0 to 1 | Share of the text VADER reads as neutral. Text in a language VADER doesn't know scores 1.0 here. |
| `vader_pos` | 0 to 1 | Share of the text VADER reads as positive. `vader_neg`, `vader_neu` and `vader_pos` add up to about 1. |
| `vader_label` | `pos` / `neu` / `neg` | `pos` if `vader_score` ≥ 0.05, `neg` if ≤ −0.05, otherwise `neu`. These are VADER's standard cutoffs. |
| `vader_claude_score` | −1 to +1 | Same as `vader_score`, but using the ratings in `claude_vader_lexicon.txt`. |
| `vader_claude_label` | `pos` / `neu` / `neg` | Same cutoffs as `vader_label`, applied to `vader_claude_score`. |
| `bert_stars` | 1 to 5 | BERT's single most likely star rating. |
| `bert_p1` … `bert_p5` | 0 to 1 | How likely BERT thinks each star rating is. The five add up to 1. |
| `bert_score` | −1 to +1 | BERT's output on the same scale as `vader_score`: the probability-weighted average star rating, mapped so that 1★ = −1, 3★ = 0 and 5★ = +1. |
| `bert_label` | `pos` / `neu` / `neg` | `pos` if `bert_score` ≥ 0.2, `neg` if ≤ −0.2, otherwise `neu`. The ±0.2 cutoff was chosen for this project, not taken from the model. |
| `siebert_p_pos` | 0 to 1 | How likely SiEBERT thinks it is that the line is positive. |
| `siebert_score` | −1 to +1 | P(positive) − P(negative). SiEBERT is usually very sure of itself, so most scores are close to −1 or +1. |
| `siebert_label` | `pos` / `neu` / `neg` | `pos` if `siebert_score` ≥ 0.5, `neg` if ≤ −0.5, otherwise `neu` (SiEBERT less than 75% sure either way). The model has no neutral class, and this cutoff was chosen for this project. |
| `roberta_p_neg`, `roberta_p_neu`, `roberta_p_pos` | 0 to 1 | How likely RoBERTa thinks it is that the line is negative, neutral or positive. The three add up to 1. |
| `roberta_score` | −1 to +1 | `roberta_p_pos` − `roberta_p_neg`. A line RoBERTa considers neutral scores near 0. |
| `roberta_label` | `pos` / `neu` / `neg` | Whichever of the three classes RoBERTa rates most likely. Unlike the other tools, this is the model's own choice, not a cutoff on the score. |
| `truncated` | `True` / `False` | Whether the line was too long for BERT, SiEBERT or RoBERTa, so their scores are based only on the beginning of the line. |
| `agree_<a>_<b>` | `True` / `False` | Whether tools `a` and `b` gave the same label, for example `agree_vader_bert` compares `vader_label` and `bert_label`. There is one column for each of the ten pairs of tools. |
| `text` | | The line's text, in full. |

### Comparing the scores

`vader_score`, `vader_claude_score`, `bert_score`, `siebert_score` and `roberta_score` share a range but measure different things, so equal numbers don't mean equal strength of sentiment. Compare their sign and their ordering across texts, not their exact values. The `agree_` columns also depend on the label cutoffs: moving BERT's ±0.2 line changes whether lukewarm texts count as agreeing. SiEBERT almost never labels a line neutral, so it disagrees with the other two on most lines they call neutral.
