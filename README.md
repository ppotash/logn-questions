# log(N)-Questions over Wikipedia Abstracts

A questioner sees *N* Wikipedia lead paragraphs and must identify a secretly
chosen target using exactly log₂(*N*) yes/no questions. An answerer sees only
the target and the question, and replies with a single word. Both roles run on
the same provider, so the game measures how well a model communicates with
*itself* across an information asymmetry.

After Potash and Suleman, [*Playing log(N)-Questions over Sentences*](https://arxiv.org/abs/1908.04660)
(arXiv:1908.04660) — ported from purpose-trained agents to pretrained frontier
models. Where the original studied whether a communication protocol could be
*learned*, this studies whether one already exists: the agents never co-adapt
and must coordinate zero-shot.

**Paper:** [`paper/main.pdf`](paper/main.pdf) · **Data:** 408 games, six models,
$363 total API spend · **Repo:** <https://github.com/ppotash/logn-questions>

---

## Results

| Model | Won | Rate | Cost | $/win | Round-1 balance |
|---|---|---|---|---|---|
| Kimi K3 | 56/68 | 82% | $68.72 | 1.23 | 49% |
| Gemini 3.8 Flash | 55/68 | 81% | $21.35 | **0.39** | 41% |
| Grok 4.6 | 51/68 | 75% | $70.02 | 1.37 | 34% |
| GPT-5.6 Sol | 49/68 | 72% | $120.60 | 2.46 | 49% |
| GLM-5.3 | 45/68 | 66% | $16.26 | **0.36** | 44% |
| Claude Opus 5 | 28/68 | 41% | $66.21 | 2.36 | 22% |

All five leading models beat Opus 5 (Fisher exact, *p* = 10⁻⁶ to 0.006). Within
the leading group only the extremes are marginally separable (*p* = 0.049); no
adjacent pair is. **The ordering inverts published general-intelligence
leaderboards** — see §8 of the paper for where independent benchmarks agree and
where they don't.

Kimi K3 won every one of its 36 games at *N* ≤ 64, the only model to play the
first five sizes without a loss.

Findings the paper argues for:

- **Win rate follows *p*^log₂*N* with a single reliability parameter.** Pooling
  the five leading models, *p* = 0.928 reproduces performance across nine set
  sizes (*r* = −0.973). The per-round independence this presumes is tested, not
  assumed: the failure rate is flat across the horizon (χ² = 10.5, df = 9,
  *p* = 0.31) and failures do not cluster. Inverting the fit, coin-flip success
  needs *p* ≥ 0.933 at ten steps and ≥ 0.986 at fifty, so the measured 0.900–0.994
  spread separates a 9-step horizon from a 115-step one.
- **Losses divide between answer errors and discrimination failures in roughly
  equal measure** — 47–55% and 42–49% depending on the judge — with 2–4%
  prediction errors. Models almost never name a document their own evidence
  excludes. Validated across three judges with a measured self-preference
  discount of 40–46% and inter-judge agreement of 97.8–98.7%.
- **The weakest model's answerer has a strong "No" bias.** Of its 34 unanimous
  answer errors, 32 are "No" answers, on properties stated in the document's
  first sentence, under an instruction that explicitly warns against defaulting
  to "No".
- **Reasoning traces grow as the candidate set shrinks**, roughly doubling from
  round 1 to round 9 at *N* = 1024, without a matching gain in reliability.
- **Two China-hosted providers refused documents in the corpus.** Moonshot
  rejected three of the 1,024 paragraphs — on Taiwan's navy, a purged Chinese
  intellectual and a detained Hong Kong activist — and Z.ai rejected the third
  of these. Located by bisection.

---

## Setup

```bash
pip install -e .
cp .env.example .env      # then add your API keys
```

Keys are read from the environment, with `.env` filling gaps. Only the
providers you actually run need keys; `logn/keys.py` checks before a run starts
rather than forty minutes into it.

## Build the corpus

```bash
python scripts/build_corpus.py --mode shards --shards 8
python scripts/build_corpus.py --verify
python scripts/inspect_pool.py --short 15
```

This reservoir-samples 4,000 lead paragraphs from English Wikipedia in one
streaming pass and freezes them, then derives nested document sets and target
assignments.

**`data/pool.jsonl` and `data/docsets/manifest.json` are committed.** They are
the artifacts that make the experiment reproducible, and rebuilding them with
`--force` changes every document in the study. Don't, unless you are starting
over.

## Run games

```bash
python scripts/run.py --dry-run
python scripts/run.py --models kimi-k3 --sizes 8 --runs 1 --budget 2   # smoke test
python scripts/run.py --models kimi-k3 --budget 90                      # full arm
```

Safe to interrupt. Completed games are skipped on re-run; aborted games and
games played under a different prompt version are replayed. Results land in
`results/raw/{model}/{N}/run{NN}.json`, one file per game, written atomically.

`--budget` is a soft ceiling: the ledger checks after each completed game, so
in-flight work finishes. At *N*=1024 that can overshoot by a few games' worth.

## Analyse

Everything below reads from `results/` and makes no API calls, except
`errors.py` when a cache is missing.

```bash
python scripts/review.py --summary                       # win rates, cost, tokens
python scripts/review.py --model kimi-k3 --size 32 --run 0 --docs
python scripts/openers.py 1024 0 1 2                     # round-1 questions side by side
python scripts/aggregation.py --round 1                  # partition quality
python scripts/signatures.py                             # answer-signature collapse
python scripts/tokens_by_round.py --balanced             # trace length by round
python scripts/by_round.py --balanced                    # agreement by round
python scripts/clustering.py --balanced                  # do errors cluster?
python scripts/answer_errors.py --model claude-opus-5 --unanimous
```

Error decomposition needs judgements. Run one judge per cache, then compare:

```bash
python scripts/errors.py --judge gemini-3.8-flash --workers 12
# rename results/adjudicated/judgments.json to judgments_gemini.json, repeat
python scripts/errors.py --all-judges                    # every cache at once
python scripts/errors.py --report --cache judgments_gpt.json
```

`errors.py` adjudicates each game's target and guess to separate answer errors
from discrimination and prediction failures. Judgements cache under
`results/adjudicated/` and are reused, so re-running is free. Use judges that
are not among the evaluated models where possible: self-preference is worth
40–46%, and `--all-judges` excludes each model's self-judgement from its own
row.

Two things about the decomposition worth knowing. It attributes each loss to
the **first** thing that went wrong, so the categories are shares of games, not
counts of errors. And a game can contain an answer error and still be won; those
are reported separately as `win+err` rather than binned with losses.

---

## Design notes

**Nested document sets.** docset(512) is a strict subset of docset(1024), so an
accuracy change between sizes is attributable to *N* rather than to one set
being easier.

**Prefix-extensible targets.** The manifest holds 64 targets per size in
bit-reversed order, so `targets[:8]` is a strict prefix of `targets[:16]` and
both are evenly stratified. Raising the run count re-uses every completed game.
No target is ever reused: a size with *N* < runs simply plays fewer games.

**Cache-aligned prompts.** Every prompt splits into system / cacheable / tail,
with documents in the cacheable segment, byte-identical across all rounds of a
game. Getting this ordering wrong costs roughly 4×. `tests/test_prompts.py`
asserts it.

**The answerer never sees document numbers.** Otherwise the game collapses into
integer bisection. It does see titles, which leaves enumeration and lexical
bisection genuinely available — observing whether models adopt them is part of
the point.

**Scoring is offline.** `game.py` computes nothing but win/loss. Everything
else is derived from logs, so metrics can be recomputed without re-spending the
API budget.

## What each result file records

Prompt version hash, schema version, corpus hash, the model ID the API echoed
back, the full question/answer history, the guess, per-call token usage
including cached and reasoning tokens, raw response text, reasoning traces,
parse mode, truncation flag, latency, and the reasoning configuration actually
used. Prompts are stored by reference — they reconstruct exactly from the
manifest plus `logn/prompts.py`, and storing them verbatim would cost ~620 MB
instead of ~26 MB.

---

## Provider notes

Six providers, six incompatible reasoning APIs. Adapters are self-healing: each
sends its best guess, drops or substitutes any parameter a 400 names, and
records the fact in `dropped_params` in every result file. **Check that field
after the first game on a new model** — a silently dropped reasoning parameter
means that arm ran without reasoning and is not comparable.

| Provider | Reasoning parameter | Notes |
|---|---|---|
| Anthropic | `thinking.type: adaptive` + `output_config.effort` | on by default; token budgets ignored |
| OpenAI | `reasoning_effort` | rejects `temperature` |
| Google | `thinkingLevel` enum | integer budget deprecated and silently ignored |
| xAI | `reasoning_effort` | reasoning tokens reported *outside* `completion_tokens` |
| Z.ai | `thinking` + `reasoning_effort` | levels are low/high/max, no medium |
| Moonshot | `reasoning_effort` | always thinks; defaults to `max` if omitted |

Three of six reject or ignore `temperature`, so those arms are not
deterministic.

**Stop-reason vocabularies are not portable.** The observed set is `stop`,
`end_turn`, `STOP`, `MAX_TOKENS`, plus `length` on OpenAI-compatible endpoints.
Matching one spelling meant a truncation guard silently never fired for one
provider and corrupted a complete arm before detection. `game.py` now matches
case-insensitively against all known spellings and warns on anything
unrecognised.

**Content filtering.** Two providers refused documents in the frozen corpus.
`scripts/kimi_probe.py` and `scripts/filter_probe.py` locate triggers by
bisection using an asymmetric
decision rule — a block is trusted immediately, a pass only after *k*
consecutive passes — because the filtering is heavily biased toward blocking
without being fully deterministic. Blocked documents are substituted per-model
via `doc_substitutions` in `config/models.yaml`, preserving *N* and the round
budget; substitutions are recorded in the affected result files.

---

## Layout

```
logn/           importable package: prompts, providers, game loop, runner
scripts/        CLI entry points and analysis
config/         experiment.yaml, models.yaml
data/           pool.jsonl and docsets/manifest.json — frozen, committed
results/raw/    one JSON per game
results/adjudicated/  judgement caches, one per judge model
paper/          main.tex, main.pdf
analysis/       ad-hoc transcript dumps (gitignored, regenerable)
```

### Scripts

| | |
|---|---|
| `build_corpus.py` | sample Wikipedia, freeze pool and doc sets |
| `inspect_pool.py` | read the corpus; `--all --run N` marks the target |
| `run.py` | play games |
| `review.py` | win rates, cost, tokens; single-game transcripts |
| `openers.py` | round-1 questions across models, side by side |
| `aggregation.py` | answer balance and information per question |
| `signatures.py` | answer-signature uniqueness and collapse |
| `errors.py` | error decomposition against one or more judges |
| `answer_errors.py` | list individual answer errors for inspection |
| `by_round.py` | agreement by round index |
| `tokens_by_round.py` | trace length by round, split by role |
| `clustering.py` | whether errors cluster within a game |
| `kimi_probe.py` | locate content-filter triggers by bisection |
| `dump_prompt.py` | reconstruct the exact prompt sent for any logged game |

`build_corpus.py` and `inspect_pool.py` import nothing from `logn`, so they run
on a bare checkout before `pip install -e .`.

## Reproducing

Every number in the paper is derivable from `results/` with the analysis
scripts above; no re-running of games is required. `scripts/errors.py --report`
regenerates the error decomposition from the cached judgments alone.

## License

Code MIT. Corpus derived from English Wikipedia (CC BY-SA 4.0). Result files
contain model outputs subject to the respective providers' terms.