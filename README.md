# llmsurgery / VEMR

Extract, inspect, modify, and reinsert transformer layers — with a safety
net. This package is built around **VEMR** (Verified Extract-Modify-Reinsert):
a protocol for editing a live language model's weights that is guaranteed to
leave the model in exactly one of two states — the edit committed, or the
original weights restored bit-for-bit — never a silently-broken third state.

Target model: GPT-2 small (`gpt2`, 12 layers, 768-wide residual stream, 3072
MLP neurons per layer), loaded via `model_loader.py`. Everything below
generalizes to any homogeneous transformer stack.

```
+-----------------------------------------------------------------+
|                        llmsurgery package                       |
|                                                                   |
|  observe --------------> understand --------------> change       |
|  (hooks, logit lens,     (circuit discovery,        (activation  |
|   neuron/attn viewers)    feature search)            editor,     |
|                                                       extractor,  |
|                                                       VEMR)       |
+-----------------------------------------------------------------+
```

---

## Table of contents

1. [Installation](#installation)
2. [Why VEMR exists](#why-vemr-exists)
3. [The core idea in one picture](#the-core-idea-in-one-picture)
4. [Algorithm 1: VEMR(M, i, E, p, D_probe, tau)](#algorithm-1-vemrm-i-e-p-d_probe-tau)
5. [Divergence signals: KL and perplexity](#divergence-signals-kl-and-perplexity)
6. [Choosing tau: the empirical sweep](#choosing-tau-the-empirical-sweep)
7. [Edit functions E(theta, p)](#edit-functions-etheta-p)
8. [Whole-layer vs. single-neuron edits](#whole-layer-vs-single-neuron-edits)
9. [Extraction, checksums, and restore](#extraction-checksums-and-restore)
10. [Whole-layer ablation vs. VEMR edits](#whole-layer-ablation-vs-vemr-edits)
11. [Supporting instruments](#supporting-instruments)
12. [Module map](#module-map)
13. [CLI reference](#cli-reference)
14. [End-to-end walkthroughs](#end-to-end-walkthroughs)
15. [Safety property](#safety-property)

---

## Installation

Defined in `pyproject.toml`: package name `llmsurgery`, requires Python
`>=3.10`, dependencies `torch`, `transformers`, `click`, built with
`setuptools`.

```bash
# from this directory
pip install -e .

# verify the CLI entry point is on PATH
llmsurgery --version
llmsurgery --help
```

`pyproject.toml` registers the console script `llmsurgery = "llmsurgery.cli:cli"`,
so once installed the `llmsurgery` command (see [CLI reference](#cli-reference))
is available directly — no need to invoke `python -m` or reference `cli.py`
by path.

---

## Why VEMR exists

Editing a transformer's weights directly (zeroing a layer, scaling it,
injecting noise, swapping in another model's layer, silencing one neuron) is
easy to *do* and easy to get *wrong*. A naive editor applies the change and
moves on — if the edit quietly wrecks the model's general behavior, nobody
finds out until much later, and there's no clean way back to "before."

VEMR closes that gap with four properties, in order:

```
  1. EXTRACT   snapshot the exact weights before touching anything
  2. MODIFY    apply the edit to a scratch copy, not the only copy
  3. REINSERT  load the edited copy into the live model  (tentative)
  4. VERIFY    measure how far the model's behavior moved
               -> within tolerance : COMMIT   (keep the edit)
               -> outside tolerance: ROLLBACK (restore the exact original)
```

Three prior parts of this project each explored one piece of this in
isolation:

| Part | File                    | Had                          | Missing                        |
|-----:|-------------------------|-------------------------------|---------------------------------|
| 6    | `activation_editor.py`  | live edits, real effect       | nothing persists past 1 forward pass |
| 8    | `layer_extractor.py`    | persisted extract/restore, checksums | no correctness check at all |
| 9    | `vemr.py`                | **both, plus** a fixed probe set, a divergence measurement, and an automatic commit/rollback decision | — |

`vemr.py`'s docstring calls this out directly: VEMR is Parts 6 and 8 "put
together, plus the piece neither had on its own."

---

## The core idea in one picture

```
                              theta_orig = deepcopy(layer.state_dict())
                                        |
                                        v
   D_probe ---> f_M(p) ---> b_before    (baseline, model UNEDITED)
                                        |
                                        |   theta_new = E(theta_orig)
                                        |   layer.load_state_dict(theta_new)
                                        v
                              [ UNSAFE WINDOW: live weights are
                                edited but not yet verified ]
                                        |
   D_probe ---> f_M(p) ---> b_after     (verify, model EDITED)
                                        |
                                        v
                    Delta = mean_p d(b_before[p], b_after[p])
                                        |
                         +--------------+--------------+
                         |                             |
                   Delta <= tau                   Delta > tau
                         |                             |
                         v                             v
                    COMMITTED                    ROLLED_BACK
              (edited weights stay live)   (layer.load_state_dict(theta_orig))
```

`D_probe` is a small, fixed, deliberately diverse prompt set — fixed on
purpose, so a Delta measurement is comparable across edits and can't be
gamed by cherry-picking easy prompts after the fact:

```python
DEFAULT_PROBE_SET = [
    "The Eiffel Tower is located in the city of",
    "My favorite food is pizza with extra cheese and",
    "The capital of Japan is",
    "Water boils at a temperature of",
    "The president of the United States lives in the",
]
```

---

## Algorithm 1: VEMR(M, i, E, p, D_probe, tau)

This is `vemr.vemr()`, implementing `algorithm-vemr.md`'s Algorithm 1 line
for line. Inputs: model `M`, target layer index `i`, edit function `E`,
divergence metric `p` (`kl` or `perplexity`), fixed probe set `D_probe`,
tolerance `tau`.

```
 1  EXTRACT     theta_orig <- deepcopy( layers(M)[i].state_dict() )
 2  BASELINE    for x in D_probe: b_before[x] <- f_M(x)
 3              (f_M(x) = softmax(final logits) for prompt x)
 4  MODIFY      theta_new <- E(theta_orig)
 5  REINSERT    layers(M)[i].load_state_dict(theta_new)     # tentative, live
 6  VERIFY      for x in D_probe: b_after[x] <- f_M(x)
 7              (model M now runs with the EDITED weights)
 8  DIVERGE     Delta <- (1/|D_probe|) * sum_x d( b_before[x], b_after[x] )
 9  DECIDE      if Delta <= tau:
10                  status <- COMMITTED          # keep theta_new live
11              else:
12                  layers(M)[i].load_state_dict(theta_orig)  # ROLLBACK
13                  status <- ROLLED_BACK
14  RETURN      audit_record{ layer_idx, tau, metric, delta, status,
                               perplexity_before/after, per-probe top-1 }
```

Python (abridged from `vemr.py`, comments trimmed):

```python
def vemr(model, tokenizer, layer_idx, edit_fn, probe_set=None, tau=0.05, metric="kl"):
    probe_set = probe_set or DEFAULT_PROBE_SET
    block = layers(model)[layer_idx]

    theta_orig = copy.deepcopy(block.state_dict())                # 1: EXTRACT

    b_before = {p: _final_distribution(model, tokenizer, p) for p in probe_set}  # 2-3
    p_before = mean(_perplexity(model, tokenizer, p) for p in probe_set)

    theta_new = edit_fn(theta_orig)                                # 4: MODIFY
    block.load_state_dict(theta_new)                               # 5: REINSERT

    try:
        b_after = {p: _final_distribution(model, tokenizer, p) for p in probe_set}  # 6-7
        p_after = mean(_perplexity(model, tokenizer, p) for p in probe_set)

        delta = mean(_kl(b_before[p], b_after[p]) for p in probe_set)  # 8

        if delta <= tau:                                          # 9-10
            status = "COMMITTED"
        else:                                                     # 11-13
            block.load_state_dict(theta_orig)
            status = "ROLLED_BACK"
    except Exception:
        block.load_state_dict(theta_orig)   # never leave weights unverified
        raise

    return audit_record
```

The `try/except` around the verify step is the implementation of the spec's
safety-property note: **no exception may escape mid-edit without guaranteeing
rollback first.** Everything between step 5 (weights go live) and the
commit/rollback decision is the "unsafe window" — the only place in the whole
protocol where the model is in a state nobody has approved yet.

---

## Divergence signals: KL and perplexity

Two independent ways to measure `d(before, after)`:

```
 KL divergence (metric="kl")
 -----------------------------------------------------------------
   f_M(x) = full softmax over the 50,257-token vocabulary,
            last-token position, for probe prompt x

   d(before, after) = sum_v  b_before[v] * log( b_before[v] / b_after[v] )

   Delta = mean over all probes in D_probe

   0.0  = identical distribution
   >0.0 = the edit measurably moved what the model would say next


 Perplexity delta (metric="perplexity")
 -----------------------------------------------------------------
   perplexity(x) = exp( cross-entropy loss of x under the model )

   Delta = | mean_perplexity_after - mean_perplexity_before |

   A simpler, single-number signal, independent of any one
   probe's top-1 answer - catches "the model got worse at
   predicting its own probe text" even if the top-1 token
   didn't flip.
```

Both are computed on the **same** probe set in the same VEMR call; `metric`
just selects which one drives the commit/rollback decision. The audit record
always reports perplexity before/after regardless of which metric gated the
decision.

---

## Choosing tau: the empirical sweep

`tau=0.05` is not a guessed constant — it's the output of a calibration
sweep (`vemr.sweep_edit_severity` + `vemr.suggest_tau`, exposed as
`llmsurgery vemr-sweep`).

```
 Step 1 - build a graded severity grid (mild / moderate / severe),
          independent of the layer it's applied to:

    mild:      scale x0.95, scale x1.05, noise std=0.001
    moderate:  scale x0.8,  scale x1.2,  noise std=0.01
    severe:    zero, scale x5.0, scale x-1.0, noise std=0.5

 Step 2 - run EVERY (layer, edit) combination through a real vemr()
          call, forced to roll back (tau=-1, so Delta > tau always) -
          this measures Delta without ever permanently changing the
          model, reusing the exact commit/rollback machinery rather
          than a parallel "just measure it" code path that could
          drift out of sync:

    for layer_idx in [2, 5, 8, 10]:
        for (severity, desc, edit_fn) in SWEEP_EDITS:
            record = vemr(model, tokenizer, layer_idx, edit_fn, tau=-1.0)
            record["delta"]  # <- captured, model is left unchanged

 Step 3 - for each candidate tau, compute:
            mild_commit_rate  = fraction of MILD   edits with delta <= tau
            severe_catch_rate = fraction of SEVERE edits with delta >  tau
          pick the tau that maximizes both at once.
```

Measured result on the default probe set, swept across layers 2/5/8/10:
mild edits topped out at delta **0.0122**; severe edits started at delta
**0.0756**. `tau = 0.05` sits in the gap between them with margin on both
sides — **100% mild-commit, 100% severe-catch** on that grid.

```
 delta
   0        0.0122                0.05                0.0756
   |----------|-------------------|--------------------|------->
   [ mild edits ]            [ tau = 0.05 ]      [ severe edits ]
      all committed             chosen here          all caught
```

`llmsurgery vemr-sweep` also runs `benchmark_vs_naive`: for each severity
tier, it compares what a **naive** policy (commit every edit blindly) would
leave the model's perplexity at, against what **VEMR** actually leaves it
at, given the chosen tau — a direct, measured demonstration that the
verification step earns its keep rather than just adding overhead.

---

## Edit functions E(theta, p)

VEMR is edit-agnostic by design: `vemr()` never inspects what `E` does
internally, only what comes out the other side. `E` is any function
`state_dict -> state_dict`.

### Whole-layer edits (`make_weight_edit_fn`)

```
 op="zero"    every weight tensor -> zeros_like(v)
 op="scale"   every weight tensor -> v * factor
 op="noise"   every weight tensor -> v + randn_like(v) * std
 op="swap"    every weight tensor -> clone of another extracted
              layer's payload (mergekit-style layer-swap,
              wrapped in VEMR's safety net instead of applied blind)
```

### Single-neuron edits (`make_neuron_edit_fn`)

GPT-2's `Conv1D` layers store weights as `(in_features, out_features)` — the
opposite convention from `nn.Linear` — so one neuron's parameters are a
**column** of `c_fc` and a **row** of `c_proj`, not a row of both:

```
  mlp.c_fc.weight    (768, 3072)   neuron n = COLUMN n
  mlp.c_fc.bias      (3072,)       neuron n = ENTRY n
  mlp.c_proj.weight  (3072, 768)   neuron n = ROW n
  mlp.c_proj.bias    (768,)        shared across ALL neurons, never touched

           768 in                    3072 hidden                768 out
   x ---> [ c_fc.weight ] --> GELU --> [ c_proj.weight ] ---> mlp_out
             (768x3072)                   (3072x768)
                 |  neuron n's                |  neuron n's
                 |  COLUMN                    |  ROW
                 v                            v
        zero this column +           zero this row
        this bias entry              (redundant once
        -> GELU(0) = 0 always        input side is 0,
        -> neuron n can never fire   but explicit > implicit)
```

```
 op="zero"    c_fc[:, n] = 0, c_fc.bias[n] = 0, c_proj[n, :] = 0
              -> pre-activation is exactly 0 for every input,
                 GELU(0)=0, neuron n can never fire again

 op="scale"   c_proj[n, :] *= factor
              -> leaves the neuron's trigger condition alone,
                 scales only its DOWNSTREAM contribution
```

This is the only edit function in the project that touches a handful of
parameters out of a layer's ~7.09M, instead of the whole state_dict at once
— what makes "find one neuron, edit exactly that one neuron, verify" a real
capstone operation rather than a whole-layer stand-in for it.

---

## Whole-layer vs. single-neuron edits

```
                    +-----------------------------+
                    |  layer i's full state_dict   |
                    |  (~7.09M params)              |
                    |                                |
   make_weight_     |  [attn.c_attn] [attn.c_proj]  |
   edit_fn touches  |  [mlp.c_fc]    [mlp.c_proj]   |
   ALL of these -->  |  [ln_1]        [ln_2]          |
                    +-----------------------------+

                    +-----------------------------+
                    |  layer i's mlp.c_fc / c_proj  |
                    |                                |
   make_neuron_     |   n0  n1  n2 ... n_k ... n3071 |
   edit_fn touches  |               ^                |
   ONLY column/row  |               |                |
   n_k          -->  |         one neuron              |
                    +-----------------------------+
```

Both flow through the exact same `vemr()` protocol — extract, baseline,
modify, reinsert, verify, commit-or-rollback. The only difference is the
shape of `E`.

---

## Extraction, checksums, and restore

`layer_extractor.py` is the primitive `vemr.py`'s safety guarantee rests on:
extract -> save -> load -> restore, proven lossless (bit-identical) via a
SHA-256 checksum over every tensor's raw bytes, in a fixed sorted key order
(dict insertion order isn't guaranteed identical across a save/load
round-trip on every platform).

```
  extract_layer(model, i)                 save_layer(payload, path)
  +-------------------------+             +----------------------+
  | model_name               |            |  torch.save(payload,  |
  | layer_idx                | --------->  |    path)               |
  | extracted_at              |            +----------------------+
  | shapes { key: shape }     |                       |
  | checksum = sha256(        |                       v
  |    sorted(state_dict) )   |             +----------------------+
  | state_dict { ... }        |  <--------  |  load_layer(path)     |
  +-------------------------+             |   = torch.load(...)   |
              |                            +----------------------+
              v
     verify_layer(payload)
     recompute checksum, compare to stored checksum
     True  = bit-identical to extraction time
     False = corrupted / truncated / silently altered

              |
              v
  restore_layer_into_model(model, i, payload, strict=True)
     shapes must match target layer i's current shapes,
     else raise (unless strict=False) - this is the raw
     mechanism `vemr()` builds its own rollback on top of.
```

`ablate_layer(model, i)` is a different operation in kind: it doesn't copy
weights out, it physically deletes block `i` from `model.transformer.h`
(an `nn.ModuleList`) — the model shrinks from N layers to N-1, in place, for
as long as that process is alive.

---

## Whole-layer ablation vs. VEMR edits

Two different product stories, benchmarked separately in `benchmark.py`:

```
 SCENARIO 1: whole-layer ablation (compression story)
 -----------------------------------------------------------------
   layer_extractor.ablate_layer(model, i)   # block physically removed

   measured before/after:  param count, latency, throughput,
                            perplexity, illustrative $/1k-forward-passes

   expectation: fewer params, faster forward pass, SOME quality
   cost - exactly what VEMR's verification step exists to catch
   if that cost is too large before an edit like this ships.


 SCENARIO 2: surgical single-neuron edit via VEMR (behavior-repair story)
 -----------------------------------------------------------------
   vemr(model, tokenizer, i, make_neuron_edit_fn(n, "zero"), tau=0.05)

   1 neuron removed out of 36,864 total MLP units (12 layers x 3072)
   -> NOT expected to move latency/memory in any way a wall-clock
      benchmark can resolve
   -> the story here is "patch one learned behavior, verified,
      without retraining," not compression
```

---

## Supporting instruments

Everything VEMR needs to decide "did the edit work / did it break anything"
is built on instruments developed earlier in the project, all sharing one
hook primitive (`hooks.py`):

```
 hooks.py
 +-----------------------------------------------------------+
 |  ActivationCapture(model)                                  |
 |    forward-hook on every block -> records each layer's     |
 |    output hidden state during one pass. cap.captured[i]    |
 |                                                              |
 |  ActivationPatcher(model, layer_idx, edit_fn)               |
 |    forward-hook that OVERWRITES one layer's output live,    |
 |    during a single forward pass only - nothing persists     |
 +-----------------------------------------------------------+
                    |                      |
      used by       |                      |     used by
      logit lens,   v                      v     circuit discovery
      neuron/attn   +---------------------------+
      viewers,      |     built on top of        |
      activation    |     hooks.py primitives     |
      editor        +---------------------------+
```

```
 logit_lens.py            reverse_lookup(model, hidden_vec):
   project any layer's       ln_f(hidden) -> lm_head -> softmax
   hidden state through       "what would the model output if it
   the model's OWN final      stopped thinking right here?"
   unembedding matrix

 neuron_viewer.py          hook block.mlp.act -> (batch, seq, 3072)
   per-neuron activation      one neuron's value at every token
   traces + probe_neuron()    position; probe_neuron() = positive
   hypothesis screening       vs. negative control prompt comparison

 attention_explorer.py     output_attentions=True (eager attn)
   per-head (seq, seq)        which tokens attend to which,
   attention matrices +       per layer AND per head (averaging
   ASCII heatmap renderer     dilutes single-head specialists)

 circuit_discovery.py      causal activation PATCHING: transplant a
   causal tracing              CLEAN run's layer-i activation into a
   (simplified ROME-style)     CORRUPTED run, see if the wrong answer
   + feature search             shifts - correlation vs. CAUSATION.
                                 discover_selective_neurons() scans
                                 all 3072 neurons in a layer at once,
                                 ranked by peak-minus-mean selectivity

 activation_diff.py        pool two prompts' neuron activations
   (mean or last-token),      (mean over tokens, or last-token only),
   rank neurons by             diff them, rank by |difference| -
   |difference|                 "what changed between these two runs"
```

These read-only instruments answer *what is the model doing and why*; VEMR
is the one component in the package that's allowed to actually change
weights, and only ever inside its own commit/rollback protocol.

---

## Module map

```
vemr/
├── __init__.py            package version
├── model_loader.py         load(), layers(), layer_state_dict() - GPT-2 access
├── hooks.py                ActivationCapture, ActivationPatcher (shared primitive)
├── logit_lens.py            Part 4: reverse_lookup, trace_prompt, trace_generation
├── neuron_viewer.py         Part 2: per-neuron activation traces, probe_neuron
├── attention_explorer.py    Part 5: per-head attention, ASCII heatmap
├── activation_editor.py     Part 6: live (non-persistent) forward-pass edits
├── activation_diff.py       neuron activation diff between two prompts
├── circuit_discovery.py     Part 7: causal patching + feature search
├── layer_extractor.py       Part 8: extract/save/load/restore/ablate + checksums
├── vemr.py                  Part 9: the VEMR algorithm itself + tau calibration
├── benchmark.py             latency/throughput/quality/cost, ablation vs. VEMR edit
└── cli.py                   `llmsurgery` - single Click entry point, one subcommand per part
```

Dependency direction is strictly downstream — `vemr.py` depends on
`model_loader.py` only; `benchmark.py` depends on `layer_extractor.py` and
`vemr.py`; `cli.py` depends on everything and adds no logic of its own
beyond argument parsing and formatting.

```
 model_loader.py  <---  hooks.py  <---  logit_lens.py, neuron_viewer.py,
      ^                                  attention_explorer.py,
      |                                  activation_editor.py,
      |                                  circuit_discovery.py,
      |                                  activation_diff.py
      |
      +----------------  layer_extractor.py  <----  vemr.py  <----  benchmark.py
                                                          ^
                                                          |
                                                       cli.py  (imports everything)
```

---

## CLI reference

Entry point: `llmsurgery` (Click group in `cli.py`, one subcommand per
project part).

| Command | Purpose |
|---|---|
| `inspect` | Print a per-layer parameter summary of the loaded model |
| `peek` | Logit-lens read-out for one prompt at one layer |
| `neurons` | Per-neuron activation trace / top-firing neurons at a token |
| `neuron-diff` | Diff two prompts' neuron activations, rank by \|difference\| |
| `logit-lens` | Full per-layer logit-lens trace for a prompt |
| `token-evolution` | Logit-lens trace for each token in a generated sequence |
| `attention` | Per-head attention weights + ASCII heatmap |
| `edit` | Live, non-persistent activation edit (Part 6) |
| `patch-sweep` | Causal activation patching across all layers |
| `feature-search` | Scan all neurons in a layer, ranked by selectivity |
| `extract` | Extract + save one layer's weights, with checksum |
| `extract-info` | Show metadata for a saved extracted-layer file |
| `restore` | Restore a saved layer's weights into a model, compare output |
| `ablate` | Physically remove a layer, compare full before/after + KL |
| **`vemr`** | **Run the full VEMR protocol on a whole layer** |
| **`vemr-sweep`** | **Calibrate tau: severity grid, commit/catch rates, VEMR vs. naive** |
| `neuron-probe` | Screen a candidate neuron against a positive/negative hypothesis |
| **`neuron-vemr`** | **Run VEMR on a single neuron, with a targeted before/after check** |
| `benchmark` | Latency/throughput/quality/cost: layer ablation vs. neuron VEMR edit |

Example — the whole protocol in one command:

```
$ llmsurgery vemr --layer 5 --op noise --std 0.01 --tau 0.05 --metric kl

VEMR: layer 5, op=noise std=0.01, metric=kl, tau=0.05

Probe set results (top-1 answer before -> after):
  'The Eiffel Tower is located in the city of'
    ' Paris' -> ' Paris'
  ...

Perplexity: 41.203 -> 41.897
Delta (kl): 0.0087  (tau=0.05)

STATUS: COMMITTED
The edit is now live in this model instance - Delta stayed within tau.
```

```
$ llmsurgery vemr --layer 5 --op zero --tau 0.05

...
Delta (kl): 4.2113  (tau=0.05)

STATUS: ROLLED_BACK
Rolled back to the exact original weights - Delta exceeded tau.
The model is bit-identical to before this command ran.
```

---

## End-to-end walkthroughs

### A. Calibrate, then edit a whole layer

```
 1. llmsurgery vemr-sweep
      -> measures Delta across a mild/moderate/severe grid,
         prints mild-commit-rate / severe-catch-rate per candidate tau
 2. pick tau from that table (default 0.05 is already calibrated)
 3. llmsurgery vemr --layer 5 --op scale --factor 0.8 --tau 0.05
      -> real edit, real verify, real commit-or-rollback
```

### B. Find, verify, and surgically fix one neuron

```
 1. llmsurgery feature-search --layer 5
      -> discover_selective_neurons: ranks all 3072 neurons by
         peak-minus-mean selectivity across a diverse prompt bank
 2. llmsurgery neuron-probe --layer 5 --neuron 792 \
        --positive "..." --negative "..."
      -> LOOKS REAL / INCONCLUSIVE hypothesis screen
 3. llmsurgery neuron-vemr --layer 5 --neuron 792 --op zero \
        --targeted-positive "..." --targeted-negative "..."
      -> BEFORE: neuron activation + real completion
      -> VEMR's general-capability verdict (commit/rollback)
      -> AFTER (only if committed): neuron activation + real completion,
         side by side with the BEFORE numbers
```

This is the full "find X, fix X, prove it worked" story:
general safety is VEMR's automatic job (nothing else in the model
should break); confirming the *targeted* change actually happened is a
separate, deliberate check the experimenter runs on purpose — VEMR has
no way to know what you intended to change, only whether the model as a
whole still behaves acceptably.

### C. Swap a layer in from another extraction (mergekit-style)

```
 1. llmsurgery extract --layer 5 --out layer5.pt
      -> payload saved with shapes + sha256 checksum
 2. llmsurgery extract-info layer5.pt
      -> inspect metadata before trusting the file
 3. llmsurgery vemr --layer 3 --op swap --swap-file layer5.pt --tau 0.05
      -> layer 3's weights tentatively replaced by layer 5's,
         verified against the probe set, committed or rolled back
```

---

## Safety property

> No exception may escape mid-edit without guaranteeing rollback first.

At every point after step 5 (weights go live, unverified) until the
commit/rollback decision at steps 9-13, the model is in a state nobody has
approved. `vemr()` wraps exactly that window in `try/except`:

```
        theta_orig captured  ---------------------------------+
                                                                 |
        block.load_state_dict(theta_new)   <-- UNSAFE WINDOW    |
                    |                            STARTS HERE    |
                    v                                            |
              try:                                               |
                  verify (probe forward passes)                  |
                  compute Delta                                  |
                  if Delta <= tau: COMMIT                         |
                  else: block.load_state_dict(theta_orig)  -------+
              except Exception:
                  block.load_state_dict(theta_orig)   <-- UNSAFE WINDOW
                  raise                                    ENDS HERE
```

Whichever path is taken — clean commit, clean rollback, or an exception
mid-verify — the function guarantees the live model ends up in exactly one
of two states: the fully-committed edit, or bit-identical original weights.
There is no code path that leaves `theta_new` live and unverified.
