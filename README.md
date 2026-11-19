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
uv run python compare_sentiment.py                  # compare each line of pg5200.txt
uv run python compare_sentiment.py other.txt        # compare each line of another file
uv run python compare_sentiment.py --json out.json  # write the results somewhere else
uv run python compare_sentiment.py --lexicon x.txt  # use another lexicon for VADER-C
```

Blank lines are skipped. The script prints a summary table and writes the results to `sentiment_comparison.json`, one row per line of text. Each run overwrites the file.

BERT, SiEBERT and RoBERTa can read at most 512 tokens (roughly 350–400 English words), so for longer lines they only see the beginning. These lines are marked with `truncated`. VADER always reads the whole line.

## Visualization

`index.html` is a static page that loads a results file and displays it. It reads the file named in the `data` URL parameter, and `sentiment_comparison.json` when there is none:

```
index.html                         # shows sentiment_comparison.json
index.html?data=pg5200-3.json      # shows another results file
```

Browsers don't let a page opened straight from disk read other files, so serve the folder, for example on GitHub Pages or locally:

```sh
python3 -m http.server     # then open http://localhost:8000/?data=sentiment_comparison.json
```

To show another text, write its results to a new file and point the page at it:

```sh
uv run python compare_sentiment.py pg5200-3.txt --json pg5200-3.json
```

The page has:

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

## Results file

A JSON object with the input file's name (`source`), the VADER-C lexicon's name (`lexicon`) and a list of `rows`, one per line of text. Each row is written on its own line of the file:

```json
{"line": 1, "text": "One morning, when Gregor Samsa woke from troubled dreams, ...", "truncated": false,
 "vader": {"score": -0.5859, "label": "neg", "neg": 0.248, "neu": 0.649, "pos": 0.103},
 "vader_claude": {"score": -0.6705, "label": "neg"},
 "bert": {"score": -0.5189, "label": "neg", "stars": 1, "probs": [0.4095, 0.3467, 0.149, 0.0614, 0.0333]},
 "siebert": {"score": -0.9958, "label": "neg", "p_pos": 0.0021},
 "roberta": {"score": -0.8371, "label": "neg", "probs": {"neg": 0.8434, "neu": 0.1503, "pos": 0.0063}}}
```

Every tool has a `score` from −1 to +1 and a `label` of `pos`, `neu` or `neg`.

| Field | Range | Meaning |
|---|---|---|
| `line` | 1 and up | Line number in the input file. |
| `text` | | The line's text, in full. |
| `truncated` | `true` / `false` | Whether the line was too long for BERT, SiEBERT or RoBERTa, so their scores are based only on the beginning of the line. |
| `vader.score` | −1 to +1 | VADER's overall score. It adds up the sentiment of individual words and squashes the total into −1…+1, so longer texts with many emotional words tend toward the ends. |
| `vader.label` | `pos` / `neu` / `neg` | `pos` if the score is ≥ 0.05, `neg` if ≤ −0.05, otherwise `neu`. These are VADER's standard cutoffs. |
| `vader.neg` | 0 to 1 | Share of the text VADER reads as negative. |
| `vader.neu` | 0 to 1 | Share of the text VADER reads as neutral. Text in a language VADER doesn't know scores 1.0 here. |
| `vader.pos` | 0 to 1 | Share of the text VADER reads as positive. `neg`, `neu` and `pos` add up to about 1. |
| `vader_claude.score` | −1 to +1 | Same as `vader.score`, but using the ratings in the VADER-C lexicon. |
| `vader_claude.label` | `pos` / `neu` / `neg` | Same cutoffs as `vader.label`. |
| `bert.score` | −1 to +1 | BERT's output on the same scale as VADER's: the probability-weighted average star rating, mapped so that 1★ = −1, 3★ = 0 and 5★ = +1. |
| `bert.label` | `pos` / `neu` / `neg` | `pos` if the score is ≥ 0.2, `neg` if ≤ −0.2, otherwise `neu`. The ±0.2 cutoff was chosen for this project, not taken from the model. |
| `bert.stars` | 1 to 5 | BERT's single most likely star rating. |
| `bert.probs` | 0 to 1 each | How likely BERT thinks each star rating is, from 1★ to 5★. The five add up to 1. |
| `siebert.score` | −1 to +1 | P(positive) − P(negative). SiEBERT is usually very sure of itself, so most scores are close to −1 or +1. |
| `siebert.label` | `pos` / `neu` / `neg` | `pos` if the score is ≥ 0.5, `neg` if ≤ −0.5, otherwise `neu` (SiEBERT less than 75% sure either way). The model has no neutral class, and this cutoff was chosen for this project. |
| `siebert.p_pos` | 0 to 1 | How likely SiEBERT thinks it is that the line is positive. |
| `roberta.score` | −1 to +1 | `probs.pos` − `probs.neg`. A line RoBERTa considers neutral scores near 0. |
| `roberta.label` | `pos` / `neu` / `neg` | Whichever of the three classes RoBERTa rates most likely. Unlike the other tools, this is the model's own choice, not a cutoff on the score. |
| `roberta.probs` | 0 to 1 each | How likely RoBERTa thinks it is that the line is negative (`neg`), neutral (`neu`) or positive (`pos`). The three add up to 1. |

### Comparing the scores

The five scores share a range but measure different things, so equal numbers don't mean equal strength of sentiment. Compare their sign and their ordering across texts, not their exact values. Agreement between labels also depends on the label cutoffs: moving BERT's ±0.2 line changes whether lukewarm texts count as agreeing. SiEBERT almost never labels a line neutral, so it disagrees with the other tools on most lines they call neutral.
