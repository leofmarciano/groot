<h1 align="center">🌳 Groot 1</h1>

<p align="center"><b>Calibrated decisions over long context, in one forward pass.</b></p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-2ea44f"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c">
  <a href="https://bend-lang.com"><img alt="Bend engine" src="https://img.shields.io/badge/engine-Bend-7c3aed"></a>
  <img alt="149M parameters" src="https://img.shields.io/badge/params-149M-555">
  <img alt="Context 64k tested" src="https://img.shields.io/badge/state-64k%20tokens-0a7ea4">
</p>

<p align="center">
  <a href="#evaluation">Evaluation</a> ·
  <a href="#from-mmbert-small-to-groot-1">Architecture</a> ·
  <a href="#start-here">Start here</a> ·
  <a href="#reproduce-the-benchmark">Reproduce</a> ·
  <a href="#limits-and-deployment-notes">Limits</a> ·
  <a href="https://huggingface.co/marcyano/Groot-1-S">🤗 Weights</a> ·
  <a href="https://github.com/leofmarciano/groot">GitHub</a>
</p>

Groot 1 is an open **typed-decision model**. It reads a state and answers typed questions about it with full probability distributions. It speaks the System One request format: `choice` among named options, ordinal `score`, and Boolean `noul`. Groot 1 S has 149.3M parameters, reads states of up to 64k tokens in our tests, and ships with a pure [Bend](https://bend-lang.com) implementation of its forward pass.

- 🎯 **Beats Jev and Julia 1 on typed decisions**: 75.2% against 73.7% and 72.7% on the full 2,000-question test split.
- 📏 **Calibrated**: typed-decision Brier of 0.072, under half of Jev's 0.156.
- 📚 **Long context**: 96–100% needle accuracy from 1k to 64k tokens, through BM25 state retrieval and a tree-reduce memory.
- 🔀 **Order-proof**: block-masked option attention makes answers exactly invariant to option order, with up to 255 options per call.
- 🇧🇷 **Multilingual**: trained on 17 public sources, 11% of them Portuguese.
- 🔌 **Drop-in**: serves the System One API, so existing `jev-latest` clients can point at it.
- ⚡ **Pure Bend engine**: the full forward pass in [Bend](https://bend-lang.com), matching PyTorch to within 1e-7.

## Evaluation

Measured on **2026-10-22** on one NVIDIA A10G with BF16 inference. Every system received the same requests built by [`scripts/benchmark.py`](scripts/benchmark.py). Accuracy is agreement between the top predicted option and the top option of the gold distribution.

| Benchmark | n | Groot 1 S | Jev | Julia 1 |
| --- | ---: | ---: | ---: | ---: |
| Typed decisions | 2,000 | **75.2%** | 73.7% | 72.7% |
| · choice | 600 | **74.2%** | 71.2% | 71.0% |
| · score | 800 | **72.9%** | 70.0% | 68.1% |
| · noul | 600 | 79.3% | **81.0%** | 80.5% |
| AG News · 4 labels | 500 | **94.2%** | 87.8% | 86.6% |
| DAIR Emotion · 6 labels | 500 | **94.8%** | 60.0% | 87.0% |
| Banking77 · 77 labels | 500 | **86.0%** | 79.0% | n/a |
| CLINC150 · 151 labels | 500 | **89.0%** | 74.4% | n/a |
| MASSIVE en · scenario, 18 labels | 233 | **91.8%** | 68.2% | 59.7% |
| MASSIVE en · intent, 60 labels | 267 | **91.8%** | 82.4% | n/a |
| MASSIVE pt · scenario, 18 labels | 256 | **91.0%** | 67.6% | 49.2% |
| MASSIVE pt · intent, 60 labels | 244 | **85.7%** | 80.3% | n/a |
| ASSIN2 pt · similarity and entailment | 500 | 68.0% | **76.4%** | 34.4% |
| HateBR pt · offensive language | 90 | 82.2% | **86.7%** | 62.2% |
| BoolQ en | 500 | 67.6% | **90.6%** | 55.2% |

Typed decisions in counts: Groot 1 S **445/600 choice, 583/800 score, 476/600 noul** (1,504/2,000). Jev 427, 560 and 486 (1,473). Julia 1 426, 545 and 483 (1,454).

**Calibration.** Lower multiclass Brier is better. It is measured against the gold distribution, which is soft for typed decisions.

| Benchmark | Groot 1 S | Jev | Julia 1 |
| --- | ---: | ---: | ---: |
| Typed decisions | **0.072** | 0.156 | 0.336 |
| AG News | **0.124** | 0.195 | 0.192 |
| DAIR Emotion | **0.074** | 0.657 | 0.222 |
| Banking77 | **0.203** | 0.318 | n/a |
| CLINC150 | **0.176** | 0.371 | n/a |
| MASSIVE en · scenario | **0.118** | 0.414 | 0.608 |
| MASSIVE pt · scenario | **0.138** | 0.440 | 0.796 |
| ASSIN2 | 0.447 | **0.330** | 1.005 |
| HateBR | 0.247 | **0.203** | 0.565 |
| BoolQ | 0.416 | **0.148** | 0.738 |

**Long context.** Needle-in-a-haystack decisions: one fact hidden in a long state of distractor text, one question per request, 100 requests per length. Needle names and templates are disjoint from training.

| State tokens | Groot 1 S accuracy | Jev accuracy | Groot 1 S p50 | Jev p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | **100%** | **100%** | **38 ms** | 78 ms |
| 4,000 | **100%** | **100%** | **52 ms** | 86 ms |
| 16,000 | 98% | **100%** | **119 ms** | 128 ms |
| 32,000 | 98% | **100%** | 236 ms | **189 ms** |
| 64,000 | 96% | **100%** | 450 ms | **360 ms** |

**Latency on short requests.** Median time for one question: Groot 1 S **34 ms**, Julia 1 **17 ms**, Jev **110 ms**. Groot and Julia run in-process on the A10G. Jev is called over its public HTTPS API from AWS us-east-2, so its time includes the network round trip.

**Protocol.**

- **Typed decisions** uses the full test split of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): 400 cases and 2,000 questions. Boolean criteria descriptions are preserved, as the dataset intends.
- **Classification sets** use 500 random test examples per dataset with seed 0. HateBR has only 90 test examples in our split.
- **Label wording.** AG News and Emotion options carry short descriptions. Banking77, CLINC150 and MASSIVE use the bare label names. Julia requires a description per option, so it receives the label name as its description.
- **n/a** marks label sets above Julia 1's native limit of 20 options.
- **Jev** was run live through `api.typesafe.ai` with the model `jev-latest` on the same requests. **Julia 1** is the public [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) checkpoint, loaded with `max_length=8192` and `head_length=512`.
- **Harness check.** Our Julia 1 typed-decisions result reproduces its own published CPU numbers: 426/600 choice and 483/600 noul in both. When Boolean descriptions are replaced by literal false/true, all three systems change. Groot rises to 76.9%, Jev to 74.0%, and Julia falls to 68.3%.
- **Contamination.** No typed-decisions test state appears in any training file. We checked by hashing every state.

[Full benchmark results](metrics/benchmark-20261022.json) · [Long-context results](metrics/needle-20261022.json)

## Where Groot is accurate — and where it can miss

Groot 1 S is strongest when the answer is **a choice grounded in the supplied state**. It leads on typed-decision choice and score questions, and on every topic, emotion and intent set we ran. Its probabilities are also the best calibrated on those sets, with typed-decision Brier under half of Jev's. It handles 151 options in one call, Portuguese as well as English, and states up to 64k tokens.

Read the classification rows with care. **Groot was trained on the train splits of every classification set in the table**, so those are in-domain supervised results. Jev and Julia 1 do not publish their training data, so the comparison is not zero-shot for Groot. The MASSIVE gap to Julia 1 is mostly a wording effect: Julia 1 reports 86.75% en and 86.25% pt on MASSIVE scenarios under its own protocol, with label descriptions.

The weak spots are **knowledge and reasoning**. BoolQ asks reading-comprehension questions whose answers often need world knowledge. There Groot reaches 67.6% against Jev's 90.6%. Groot also trails Jev on ASSIN2 sentence similarity and on HateBR. On typed-decision Boolean questions with descriptive criteria it is 1.7 points behind Jev. That gap appears because Groot was trained with literal false/true labels; the converter now keeps the descriptions for the next training run.

Long states cost accuracy and time. Needle retrieval drops to 96% at 64k tokens, where Jev stays at 100%. Above 16k tokens Groot is slower than Jev's API. On validation typed decisions padded with 4k to 32k tokens of distractor text, accuracy is 61.5%. The same model scores 76.9% on short typed decisions in that format. Evaluate on your own questions and states before you rely on it.

## Why a decision model?

Many product decisions are a finite choice: route a ticket, rate a reply, flag a trace, approve an invoice. A generative model can make them, but it costs a full decode per answer and gives no usable probability. A decision model reads the state and the candidate answers together and returns a distribution in one forward pass.

Groot answers many questions about one state in a single batch, and the state is encoded once. Options are scored as a set, so **their order cannot change the answer**; the tests check this exactly. Per-type temperatures are fitted on validation data, and a separate head estimates the probability that the answer is correct.

## From mmBERT-small to Groot 1

Groot 1 S starts from [JHU CLSP's mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), a multilingual ModernBERT encoder. It keeps that encoder and tokenizer and adds a long-state memory, option fusion layers and decision heads.

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

| | mmBERT-small | Groot 1 S |
| --- | --- | --- |
| Main interface | Encoder features, masked-token modeling | `state` + named typed questions → distributions and answers |
| Parameters | About 140M | 149.3M; 98.3M of them are the 256k-token vocabulary embeddings |
| Context | 8,192 tokens | State of up to 512 chunks of 512 tokens; tested to 64k. Question window 1,024 tokens, 48 tokens per option |
| Options | n/a | `choice` 1–255, `score` 1–10 levels, `noul` |
| Output | Token features | Probabilities, choice or expected score, confidence, p(correct) |

**Training.** Groot 1 S was trained on four A10G GPUs in three stages, then calibrated.

1. **Generalist.** Two epochs over 852k rows from 17 public sources, 11% of them in Portuguese.
2. **Specialist.** Eight epochs on the typed-decisions train split, 5,400 questions split by case. The released weights are a 0.7/0.3 average of the specialist and the generalist.
3. **Long context.** One epoch with retrieval on: synthetic needle states, typed decisions padded to 4k–32k tokens with distractors, and replay of short data.

## Bend engine

[`bend/`](bend/) holds the complete forward pass in [Bend](https://bend-lang.com): kernels, the block-masked encoder, the tree memory and the fusion layers. It parallelizes over chunks and questions with fork/join. It runs as JavaScript through the generated host, or as a native binary on CPU threads and GPU. On the test fixture both paths match PyTorch to within 1e-7.

```bash
groot-export bend --model Groot-1-S --out groot-1-s.gbw   # flat f32 weights
bend bend/groot/host_api.bend -o bend/groot/host_api.mjs              # JS host
node bend/groot/parity.mjs                                            # parity vs PyTorch
bend bend/main.bend -o groot                                          # native binary
```

The Python serializer still builds the token request blob. The PyTorch runtime remains the fast path; the Bend engine has not been benchmarked at full model size. An ONNX export is not yet available for this architecture.

## Start here

Python 3.10+ is required. The [Hugging Face repository](https://huggingface.co/marcyano/Groot-1-S) holds the 569 MiB checkpoint together with the code. Download it and install the package:

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('marcyano/Groot-1-S', local_dir='Groot-1-S')"
python -m pip install -e ./Groot-1-S            # inference and training
python -m pip install -e './Groot-1-S[serve]'   # + HTTP server
```

The code alone is also on [GitHub](https://github.com/leofmarciano/groot).

Keep the engine loaded between requests:

```python
from groot1.schema import SystemOneRequest
from groot1.serve import GrootEngine

engine = GrootEngine("Groot-1-S")   # cuda when available, else cpu

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

The state can be a string or any JSON value. Question names are returned unchanged, and all questions share one encoding of the state.

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Mapping of 1–255 IDs to descriptions, which may be `null` | `choice`, `confidence`, `probabilities` keyed by ID |
| `score` | Ordered list of 1–10 level descriptions | `score` as the expected zero-based level, `confidence`, `legend`, `probabilities` |
| `noul` | Optional `false` and `true` descriptions | `noul`, the probability of true |

Serve the same API over HTTP:

```bash
groot-serve --model Groot-1-S --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' \
  -d '{"model": "groot-latest", "state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}}'
```

`POST /v1/systemone` accepts `groot-latest`, the checkpoint name and `jev-latest`, so existing System One clients can point at it. The header `groot-extended: true` adds `p_correct` and truncation flags per answer. `GET /v1/models` and `GET /healthz` are also available.

### Reproduce the benchmark

```bash
groot-data build --out data/v1                         # public sources, see groot1/data.py
python scripts/benchmark.py build --test data/v1/test.jsonl --per-source 500 --out bench.jsonl \
    --sources typed_decisions agnews emotion banking77 clinc150 massive_en massive_pt assin2 hatebr boolq
python scripts/benchmark.py run --rows bench.jsonl --groot Groot-1-S \
    --julia path/to/Julia-1 --jev --out results.json   # Jev needs TYPESAFE_API_KEY
python scripts/benchmark.py table --results results.json
```

Each system is optional. Omit `--julia` or `--jev` to skip it. Long-context accuracy comes from `scripts/needle_eval.py` on needle sets made by `python -m groot1.longctx needles`.

### Train

```bash
groot-data build --out data/v1
groot-data mix --src data/v1 --out data/mix               # caps per source, stratified val/test
torchrun --nproc_per_node 4 -m groot1.train --preset groot-1-s --train data/mix/train.jsonl --val data/mix/val.jsonl \
    --out runs/s1 --epochs 2 --lr 3e-5 --head-lr 1e-4 --batch-size 64 --grad-checkpoint --max-tokens 24000 --max-attn 12e6
```

The trainer batches by token budget, resumes from periodic checkpoints and rolls back on loss spikes. It fits per-type temperatures at the end of the run. Use `--init` to continue from a checkpoint and `--set key=value` to override configuration, for example `--set state_retrieval=true`. `scripts/soup.py` averages checkpoints. Run the tests with `pytest tests/`.

## Limits and deployment notes

- The FP32 checkpoint is **569 MiB**. CPU inference works; a GPU is needed for long states at interactive latency.
- Classification results are **in-domain**: Groot saw the train splits of those datasets. Expect lower accuracy on new label sets and domains.
- Groot is weak on questions that need world knowledge or multi-step reasoning, as BoolQ shows.
- Surface cues can outweigh meaning. "I was charged twice for the same order" routes to shipping with 0.89 probability, because the word "order" dominates. Keep option descriptions distinct and test your own phrasing.
- States above 16k tokens are slower than Jev's API, and accuracy degrades with long distractor-heavy states.
- Images, `box` questions and other System One extensions are not supported; such requests are rejected.
- The probabilities are calibrated on validation data, but they are not guarantees. Evaluate on your own workflow before using Groot for consequential actions.

The code and model artifacts are licensed under **Apache 2.0**. mmBERT is MIT-licensed. Each training dataset keeps its own license; see [NOTICE](NOTICE).

---

<p align="center">Marciano Theory SI</p>
