<h1 align="center">🌳 Groot 1</h1>

<p align="center"><b>Calibrated decisions over long context, in one forward pass.</b></p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-2ea44f"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c">
  <a href="https://bend-lang.com"><img alt="Bend engine" src="https://img.shields.io/badge/engine-Bend-7c3aed"></a>
  <img alt="Groot 1: 482M parameters, Groot 1 S: 149M" src="https://img.shields.io/badge/params-482M%20%7C%20149M-555">
  <img alt="Context 64k tested" src="https://img.shields.io/badge/state-64k%20tokens-0a7ea4">
</p>

<p align="center">
  <a href="#evaluation">Evaluation</a> ·
  <a href="#the-folha-correction">Folha correction</a> ·
  <a href="#from-mmbert-to-groot-1">Architecture</a> ·
  <a href="#start-here">Start here</a> ·
  <a href="#reproduce-the-benchmark">Reproduce</a> ·
  <a href="#limits-and-deployment-notes">Limits</a> ·
  <a href="https://huggingface.co/marcyano/Groot-1">🤗 Groot 1</a> ·
  <a href="https://huggingface.co/marcyano/Groot-1-S">🤗 Groot 1 S</a> ·
  <a href="https://github.com/leofmarciano/groot">GitHub</a>
</p>

Groot 1 is an open **typed-decision model**. It reads a state and answers typed questions about it with full probability distributions. It speaks the System One request format: `choice` among named options, ordinal `score`, and Boolean `noul`. It ships in two sizes and with a pure [Bend](https://bend-lang.com) implementation of the forward pass.

- **Groot 1** averages the probabilities of two members: an S model on mmBERT-small and an M model on mmBERT-base. It has 482M parameters in total and is the most accurate release.
- **Groot 1 S** is the single 149.3M model. It runs at half the latency of Groot 1.

Highlights:

- 📰 **Beats Jev on zero-shot Portuguese news**: Groot 1 scores 87.8% on 400 Folha de S.Paulo articles, against 85.8% for Jev. None of these articles, or any Folha text, was in training.
- 🎯 **Beats Jev and Julia 1 on typed decisions**: Groot 1 scores 75.3% on the full 2,000-question test split, against 73.7% for Jev and 72.7% for Julia 1.
- 📏 **Calibrated**: Groot 1 has a typed-decision Brier of 0.066, under half of Jev's 0.156. On Folha it is 0.192, against 0.215 for Jev.
- 📚 **Long context**: 96–100% needle accuracy from 1k to 64k tokens, through BM25 state retrieval and a tree-reduce memory.
- 🔀 **Order-proof**: block-masked option attention makes answers exactly invariant to option order, with up to 255 options per call.
- 🇧🇷 **Multilingual**: trained on public sources in English and Portuguese, plus Portuguese web news labelled by URL section.
- 🔌 **Drop-in**: serves the System One API, so existing `jev-latest` clients can point at it.
- ⚡ **Pure Bend engine**: the full forward pass of Groot 1 S in [Bend](https://bend-lang.com), matching PyTorch to within 1e-7.

> **v1.1, 2026-10-23.** Groot 1 S v1.0 scored 33.8% on the Folha news test. Its answers collapsed onto a few labels it had memorised. This release fixes that failure. [What failed and how it was fixed](#the-folha-correction).

## Evaluation

Measured on **2026-10-23** on one NVIDIA A10G with BF16 inference. Every system received the same requests built by [`scripts/benchmark.py`](scripts/benchmark.py). Accuracy is agreement between the top predicted option and the top option of the gold distribution.

| Benchmark | n | Groot 1 | Groot 1 S | Jev | Julia 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Folha news pt · zero-shot, 8 sections | 400 | **87.8%** | 84.5% | 85.8% | 49.5% |
| Typed decisions | 2,000 | **75.3%** | 75.2% | 73.7% | 72.7% |
| · choice | 600 | **73.7%** | **73.7%** | 71.2% | 71.0% |
| · score | 800 | **72.8%** | 71.5% | 70.0% | 68.1% |
| · noul | 600 | 80.3% | **81.7%** | 81.0% | 80.5% |
| AG News · 4 labels | 500 | **92.8%** | **92.8%** | 87.8% | 86.6% |
| DAIR Emotion · 6 labels | 500 | **94.4%** | 94.0% | 60.0% | 87.0% |
| Banking77 · 77 labels | 500 | 84.0% | **84.6%** | 79.0% | n/a |
| CLINC150 · 151 labels | 500 | 84.6% | **87.2%** | 74.4% | n/a |
| MASSIVE en · scenario, 18 labels | 233 | **91.4%** | **91.4%** | 68.2% | 59.7% |
| MASSIVE en · intent, 60 labels | 267 | **89.5%** | 89.1% | 82.4% | n/a |
| MASSIVE pt · scenario, 18 labels | 256 | **89.1%** | 87.5% | 67.6% | 49.2% |
| MASSIVE pt · intent, 60 labels | 244 | **86.1%** | 84.0% | 80.3% | n/a |
| ASSIN2 pt · similarity and entailment | 500 | 70.2% | 70.0% | **76.4%** | 34.4% |
| HateBR pt · offensive language | 90 | 83.3% | 84.4% | **86.7%** | 62.2% |
| BoolQ en | 500 | 72.0% | 69.0% | **90.6%** | 55.2% |

Counts on the two headline sets:

- **Folha news.** Groot 1 351/400, Groot 1 S 338/400, Jev 343/400, Julia 1 198/400.
- **Typed decisions.** Groot 1 got 442/600 choice, 582/800 score and 482/600 noul, for 1,506/2,000. Groot 1 S got 1,504/2,000, Jev 1,473 and Julia 1 1,454.

**Calibration.** Lower multiclass Brier is better. It is measured against the gold distribution, which is soft for typed decisions.

| Benchmark | Groot 1 | Groot 1 S | Jev | Julia 1 |
| --- | ---: | ---: | ---: | ---: |
| Folha news | **0.192** | 0.228 | 0.215 | 0.768 |
| Typed decisions | **0.066** | 0.068 | 0.156 | 0.336 |
| AG News | 0.116 | **0.108** | 0.195 | 0.192 |
| DAIR Emotion | 0.091 | **0.072** | 0.657 | 0.222 |
| Banking77 | 0.254 | **0.252** | 0.318 | n/a |
| CLINC150 | 0.229 | **0.227** | 0.371 | n/a |
| MASSIVE en · scenario | **0.120** | 0.145 | 0.414 | 0.608 |
| MASSIVE pt · scenario | **0.147** | 0.190 | 0.440 | 0.796 |
| ASSIN2 | 0.391 | 0.384 | **0.330** | 1.005 |
| HateBR | 0.230 | 0.223 | **0.203** | 0.565 |
| BoolQ | 0.393 | 0.388 | **0.148** | 0.738 |

**Long context.** Needle-in-a-haystack decisions. Each request hides one fact in a long state of distractor text and asks one question. There are 100 requests per length. Needle names and templates are disjoint from training.

| State tokens | Groot 1 | Groot 1 S | Jev | Groot 1 p50 | Groot 1 S p50 | Jev p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | **100%** | **100%** | **100%** | 85 ms | **40 ms** | 78 ms |
| 4,000 | **100%** | 99% | **100%** | 128 ms | **53 ms** | 86 ms |
| 16,000 | 98% | 98% | **100%** | 313 ms | **120 ms** | 128 ms |
| 32,000 | 98% | 99% | **100%** | 616 ms | 239 ms | **189 ms** |
| 64,000 | 96% | 96% | **100%** | 1,180 ms | 451 ms | **360 ms** |

**Latency on short requests.** Median time for one question:

| System | p50 |
| --- | ---: |
| Julia 1 | **17 ms** |
| Groot 1 S | 36 ms |
| Groot 1 | 73 ms |
| Jev | 110 ms |

Groot and Julia run in-process on the A10G. Jev is called over its public HTTPS API from AWS us-east-2, so its time includes the network round trip.

**Protocol.**

- **Folha news** has 400 articles from [emdemor/news-of-the-brazilian-newspaper](https://huggingface.co/datasets/emdemor/news-of-the-brazilian-newspaper), 50 for each of 8 newsroom sections: poder, mercado, mundo, esporte, cotidiano, ilustrada, tec and equilibrioesaude. Every section is offered with a one-line Portuguese description. The gold label is the section the newsroom filed the article under.
- **Typed decisions** uses the full test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): 400 cases and 2,000 questions. Boolean criteria descriptions are preserved, as the dataset intends.
- **Classification sets** use 500 random test examples per dataset with seed 0. HateBR has only 90 test examples in our split.
- **Label wording.** AG News and Emotion options carry short descriptions. Banking77, CLINC150 and MASSIVE use the bare label names. Julia requires a description per option, so it receives the label name as its description.
- **n/a** marks label sets above Julia 1's native limit of 20 options.
- **Systems.**
  - **Jev** was run live through `api.typesafe.ai` with the model `jev-latest` on the same requests.
  - **Julia 1** is the public [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) checkpoint, loaded with `max_length=8192` and `head_length=512`.
  - Jev and Julia 1 results are from 2026-10-22.
- **Harness check.** Our Julia 1 typed-decisions result reproduces its own published CPU numbers: 426/600 choice and 483/600 noul in both.
- **Contamination.** No typed-decisions test state appears in any training file. We checked by hashing every state. No Folha text was used for training: the Folha datasets, including the IARA news corpus, are excluded, and Folha and UOL domains are blocked in the web-news data.

[Full benchmark results](metrics/benchmark-20261023.json) · [Long-context results](metrics/needle-20261023.json) · [v1.0 results](metrics/benchmark-20261022.json)

## The Folha correction

**What failed.** After the v1.0 launch, a user ran Groot 1 S on Portuguese news and got worse results than Jev. We rebuilt the test on 400 Folha de S.Paulo articles and confirmed it. v1.0 scored **33.8%**, against 85.8% for Jev and 49.5% for Julia 1.

Its answers collapsed onto a few section names. v1.0 had learned its classification skills from fixed label sets, and it had learned the priors of those label names along with them. Offered a new set of sections, it picked the names it had memorised rather than reading the article.

**How it was fixed.** Three changes, all trained on public data only:

1. **Label-space diversification.**
   - Topic data comes from 11 public sets, among them DBpedia, Yahoo Answers, HuffPost, NYT, SIB-200, BBC and two Portuguese news sets.
   - Every set was rebuilt with random label subsets, keys and descriptions in English or Portuguese, neutral IDs, and varied question wording.
   - The model can no longer win by remembering label names.
2. **Portuguese web news labelled by URL section.**
   - Articles come from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `por_Latn`.
   - Labels come from URL paths and subdomains such as `/politica/` or `esporte.`, plus a few domain rules.
   - Folha and UOL are blocked.
3. **Teacher soft labels and a larger model.**
   - About 20k topic rows carry the next-token distribution of the open [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), blended 50/50 with the gold label.
   - A new **M** model starts from mmBERT-base, and Groot 1 averages it with S.
   - Jev outputs were never used for training.

**Selection protocol.**

- **Development set.** Model selection used a separate set of 400 Folha articles, drawn from the same shuffle as the test but with no overlap. It was combined with the typed-decisions validation split.
- **Test runs.** The test was run for v1.0 and then for three final candidates, two Groot 1 S checkpoints (84.5% and 84.3%) and this Groot 1 ensemble (87.8%). Nothing else saw it.
- **How sure the lead is.**
  - Groot 1's lead over Jev is 8 articles out of 400. With 400 articles, each score carries a margin of about ±3 points.
  - Read Groot 1 as on par with Jev or better on this test, not decisively better.
  - Groot 1 S alone is 1.3 points behind Jev.

## Where Groot is accurate — and where it can miss

Groot is strongest when the answer is **a choice grounded in the supplied state**. It leads on typed-decision choice and score questions, on zero-shot Folha news, and on every topic, emotion and intent set we ran. Its probabilities are also the best calibrated on those sets. Its typed-decision Brier is under half of Jev's. It handles 151 options in one call, Portuguese as well as English, and states up to 64k tokens.

Read the classification rows with care. **Groot was trained on the train splits of every classification set in the table**, so those are in-domain supervised results. Jev and Julia 1 do not publish their training data, so the comparison is not zero-shot for Groot. Folha news is the zero-shot check.

v1.1 trades a little in-domain accuracy for that robustness. On CLINC150, Groot 1 S fell from 89.0% to 87.2%, and on AG News from 94.2% to 92.8%.

The weak spots are **knowledge and reasoning**:

- **BoolQ.** These reading-comprehension questions often need world knowledge. Groot 1 reaches 72.0%, against Jev's 90.6%.
- **ASSIN2 and HateBR.** Groot also trails Jev on sentence similarity and on offensive-language detection.

Long states cost accuracy and time. Needle retrieval drops to 96% at 64k tokens, where Jev stays at 100%. Above 16k tokens Groot is slower than Jev's API, and Groot 1 runs two models, so it is slower still. Evaluate on your own questions and states before you rely on it.

## Why a decision model?

Many product decisions are a finite choice: route a ticket, rate a reply, flag a trace, approve an invoice. A generative model can make them, but it costs a full decode per answer and gives no usable probability. A decision model reads the state and the candidate answers together and returns a distribution in one forward pass.

Groot answers many questions about one state in a single batch, and the state is encoded once. Options are scored as a set, so **their order cannot change the answer**; the tests check this exactly. Per-type temperatures are fitted on validation data, and a separate head estimates the probability that the answer is correct.

## From mmBERT to Groot 1

Groot 1 S starts from [JHU CLSP's mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), a multilingual ModernBERT encoder. Groot 1 M starts from [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base). Both keep the encoder and tokenizer and add a long-state memory, option fusion layers and decision heads. Groot 1 runs S and M on the same request and averages their probabilities with equal weights.

```
                       ┌──────────────── long state ────────────────┐
state ──► 512-token chunks ──► encoder ──► chunk summaries ──► tree reduce ──► state memory (K/V)
   │                                                                                │
   └─► head of the state + BM25-retrieved pieces (512 tokens)                       │
                     │                                                              │
[CLS] question [SEP] state window [SEP] [MASK] option 1 … [MASK] option K           │
                     │                                                              │
     one encoder pass; every option attends to the prefix and itself only           │
                     │                                                              │
        option vectors ──► 3 fusion layers: self-attention over options ◄── cross-attention
                     │
     scorer + per-type temperature ──► probabilities, answer, confidence, p(correct)
```

- **Early fusion with block masks.** The question, a window of the state and every option go through the encoder together. A block-restricted attention mask lets each option see the prefix and itself only. All options share one position, so the model is exactly invariant to their order.
- **State retrieval.** Long states keep their first 64 tokens in the window. The rest of the window is filled with the 64-token pieces that best match the question and options under BM25, in their original order.
- **Tree-reduce memory.** The whole state is also encoded in 512-token chunks and merged pairwise into a tree. The fusion layers cross-attend to every chunk and tree node, so evidence outside the window still reaches the decision.
- **Typed heads.** One scorer covers all three types. Calibration uses per-type temperatures, and the confidence follows the System One formulas.

| | mmBERT-small | Groot 1 S | Groot 1 |
| --- | --- | --- | --- |
| Main interface | Encoder features, masked-token modeling | `state` + named typed questions → distributions and answers | Same as Groot 1 S |
| Parameters | About 140M | 149.3M; 98.3M of them are the 256k-token vocabulary embeddings | 482M: S plus a 333M M member on mmBERT-base |
| Context | 8,192 tokens | State of up to 512 chunks of 512 tokens; tested to 64k. Question window 1,024 tokens, 48 tokens per option | Same as Groot 1 S |
| Options | n/a | `choice` 1–255, `score` 1–10 levels, `noul` | Same as Groot 1 S |
| Output | Token features | Probabilities, choice or expected score, confidence, p(correct) | Member probabilities and p(correct), averaged |

**Training.** Both models were trained on four A10G GPUs.

v1.0 of Groot 1 S was trained in three stages, then calibrated:

1. **Generalist.** Two epochs over 852k rows from 17 public sources, 11% of them in Portuguese.
2. **Specialist.** Eight epochs on the typed-decisions train split, 5,400 questions split by case. The v1.0 weights are a 0.7/0.3 average of the specialist and the generalist.
3. **Long context.** One epoch with retrieval on, using three data sources:
   - synthetic needle states;
   - typed decisions padded to 4k–32k tokens with distractors;
   - replay of short data.

v1.1 adds the zero-shot stages described in [the Folha correction](#the-folha-correction).

- **Groot 1 S v1.1** continues from v1.0 on diversified topic data, URL-labelled Portuguese news, teacher soft labels, typed-decision retention data and needle replay. The released weights average the checkpoints of the last stage, and per-type temperatures are refitted afterwards.
- **Groot 1 M** first trains for one generalist epoch over 872k rows. It then fine-tunes on 224k rows of the v1.1 mixture.
- **Groot 1** pairs an S checkpoint from a sibling run of the last S stage with an early checkpoint of the M fine-tune. Both members were chosen on the development set.

## Bend engine

[`bend/`](bend/) holds the complete forward pass of Groot 1 S in [Bend](https://bend-lang.com): kernels, the block-masked encoder, the tree memory and the fusion layers. It parallelizes over chunks and questions with fork/join. It runs as JavaScript through the generated host, or as a native binary on CPU threads and GPU. On the test fixture both paths match PyTorch to within 1e-7.

```bash
groot-export bend --model Groot-1-S --out groot-1-s.gbw   # flat f32 weights
bend bend/groot/host_api.bend -o bend/groot/host_api.mjs              # JS host
node bend/groot/parity.mjs                                            # parity vs PyTorch
bend bend/main.bend -o groot                                          # native binary
```

The Python serializer still builds the token request blob. The PyTorch runtime remains the fast path, and the Bend engine has not been benchmarked at full model size. The M member and ensemble averaging are not yet parity-tested in Bend. An ONNX export is not yet available for this architecture.

## Start here

Python 3.10+ is required. Each Hugging Face repository holds the checkpoint together with the code:

- [Groot 1](https://huggingface.co/marcyano/Groot-1) is 1.9 GB. It contains an `ensemble.json` plus the `s/` and `m/` members.
- [Groot 1 S](https://huggingface.co/marcyano/Groot-1-S) is 569 MiB.

Download one and install the package:

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('marcyano/Groot-1', local_dir='Groot-1')"
python -m pip install -e ./Groot-1            # inference and training
python -m pip install -e './Groot-1[serve]'   # + HTTP server
```

Use `marcyano/Groot-1-S` and `Groot-1-S` instead for the single small model.

The code alone is also on [GitHub](https://github.com/leofmarciano/groot).

Keep the engine loaded between requests:

```python
from groot1.schema import SystemOneRequest
from groot1.serve import GrootEngine

engine = GrootEngine("Groot-1")   # or "Groot-1-S"; cuda when available, else cpu

req = SystemOneRequest(
    state={"ticket": "I can't log in since I reset my password yesterday.", "customer_tier": "gold"},
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "Billing and payment disputes",
                "shipping": "Shipping and delivery",
                "access": "Account access and login",
            },
        },
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "The customer should be refunded.",
                   "criteria": {"false": "No refund is due.", "true": "A refund is due."}},
    },
)
res = engine.answer(req.state, req.questions)
print(res.answers["team"].choice, res.answers["team"].probabilities)
print(res.answers["urgency"].score, res.answers["refund"].noul)
```

The state can be a string or any JSON value. Question names are returned unchanged, and all questions share one encoding of the state per model. A checkpoint directory with an `ensemble.json` file loads as an ensemble. The file lists member directories and optional weights. Any set of Groot checkpoints can be combined this way.

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Mapping of 1–255 IDs to descriptions, which may be `null` | `choice`, `confidence`, `probabilities` keyed by ID |
| `score` | Ordered list of 1–10 level descriptions | `score` as the expected zero-based level, `confidence`, `legend`, `probabilities` |
| `noul` | Optional `false` and `true` descriptions | `noul`, the probability of true |

Serve the same API over HTTP:

```bash
groot-serve --model Groot-1 --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"model": "groot-latest", "state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}}'
```

`POST /v1/systemone` accepts `groot-latest`, the checkpoint name and `jev-latest`, so existing System One clients can point at it. The header `groot-extended: true` adds `p_correct` and truncation flags per answer. `GET /v1/models` and `GET /healthz` are also available.

### Reproduce the benchmark

```bash
groot-data build --out data/v1                         # public sources, see groot1/data.py
python scripts/benchmark.py build --test data/v1/test.jsonl --per-source 500 --out bench.jsonl \
    --sources typed_decisions agnews emotion banking77 clinc150 massive_en massive_pt assin2 hatebr boolq
python scripts/benchmark.py run --rows bench.jsonl --groot Groot-1 \
    --julia path/to/Julia-1 --jev --out results.json   # Jev needs TYPESAFE_API_KEY
python scripts/benchmark.py table --results results.json
```

Each system is optional. Omit `--julia` or `--jev` to skip it. Long-context accuracy comes from `scripts/needle_eval.py` on needle sets made by `python -m groot1.longctx needles`.

The v1.1 training data adds two builders:

- `python -m groot1.topics build` builds the diversified topic sets.
- `python -m groot1.topics webnews` builds URL-labelled news from FineWeb-2 extracts made by `scripts/webnews*.py`.

The Folha test comes from `emdemor/news-of-the-brazilian-newspaper`: 50 articles per section after a seed-0 shuffle. The development set uses articles 51–100.

### Train

```bash
groot-data build --out data/v1
groot-data mix --src data/v1 --out data/mix               # caps per source, stratified val/test
torchrun --nproc_per_node 4 -m groot1.train --preset groot-1-s --train data/mix/train.jsonl --val data/mix/val.jsonl \
    --out runs/s1 --epochs 2 --lr 3e-5 --head-lr 1e-4 --batch-size 64 --grad-checkpoint --max-tokens 24000 --max-attn 12e6
```

The trainer batches by token budget, resumes from periodic checkpoints and rolls back on loss spikes. It fits per-type temperatures at the end of the run. Use `--init` to continue from a checkpoint and `--set key=value` to override configuration, for example `--set state_retrieval=true`. `scripts/soup.py` averages checkpoints. Run the tests with `pytest tests/`.

## Limits and deployment notes

- Groot 1 is **1.9 GB** and runs two models per request. Groot 1 S is **569 MiB** at half the latency. CPU inference works. A GPU is needed for long states at interactive latency.
- Classification results are **in-domain**: Groot saw the train splits of those datasets.
- Folha news is zero-shot, but it is one newspaper and 8 sections. Expect lower accuracy on unusual label sets and domains.
- Groot is weak on questions that need world knowledge or multi-step reasoning, as BoolQ shows.
- Surface cues can outweigh meaning. "I was charged twice for the same order" routes to shipping, because the word "order" dominates. Groot 1 S gives it 0.97 probability and Groot 1 gives it 0.56. Keep option descriptions distinct and test your own phrasing.
- States above 16k tokens are slower than Jev's API, and accuracy degrades with long distractor-heavy states.
- Images, `box` questions and other System One extensions are not supported; such requests are rejected.
- The probabilities are calibrated on validation data, but they are not guarantees. Evaluate on your own workflow before using Groot for consequential actions.

The code and model artifacts are licensed under **Apache 2.0**. mmBERT is MIT-licensed. Each training dataset keeps its own license; see [NOTICE](NOTICE).

---

<p align="center">Marciano Theory SI</p>
