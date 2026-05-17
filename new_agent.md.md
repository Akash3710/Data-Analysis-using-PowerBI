# AutoSLM — Copilot Agent Instructions

You are an autonomous coding and research agent.

When given this file, you must:
1. Read every section completely before writing a single line of code
2. Create the entire project from scratch — every file, every folder
3. Verify the project runs correctly end-to-end
4. Then begin the autonomous optimization loop

Do not ask for clarification. Do not wait for approval between steps.
Execute everything described here autonomously and sequentially.

---

## Phase 1 — Bootstrap the Project

### Step 1.1 — Create the folder structure

Create exactly this directory layout in the current working directory:

```
autoslm/
├── config/
├── data/
├── experiments/
├── tokenizer/
├── logs/
├── checkpoints/
└── src/
```

### Step 1.2 — Create requirements.txt

Create `autoslm/requirements.txt` with these dependencies:

```
torch
tokenizers
datasets
numpy
scikit-learn
pyyaml
tqdm
psutil
```

### Step 1.3 — Install dependencies

Run:

```bash
pip install -r autoslm/requirements.txt
```

---

## Phase 2 — Create All Config Files

### File: `config/hardware.yaml`

```yaml
device: cpu               # Options: cpu | cuda | mps — auto-detected at runtime
precision: float32        # Options: float32 | float16 | bfloat16
max_ram_usage_gb: 14
num_workers: 0            # Set > 0 only for GPU environments
```

### File: `config/runtime.yaml`

```yaml
max_experiment_minutes: 10
val_stagnation_patience: 5
f1_stagnation_patience: 3
nan_tolerance: 0
eval_every_n_steps: 50
log_every_n_steps: 10
checkpoint_every_n_steps: 100
```

### File: `config/baseline.yaml`

```yaml
# Architecture
hidden_size: 256
layers: 4
heads: 4
ffn_multiplier: 4
seq_length: 128
vocab_size: 8000
activation: swiglu        # Options: swiglu | gelu | relu
normalization: rmsnorm    # Options: rmsnorm | layernorm
positional_embedding: rope
dropout: 0.1

# Training
learning_rate: 3.0e-4
optimizer: adamw          # Options: adamw | sgd
scheduler: cosine         # Options: cosine | linear | constant
warmup_ratio: 0.1
batch_size: 4
gradient_accumulation: 8
weight_decay: 0.01
gradient_clipping: 1.0

# Tokenizer
tokenizer_type: bpe       # Options: bpe | wordpiece | unigram
```

### File: `config/search_space.yaml`

```yaml
architecture:
  hidden_size: [128, 192, 256, 320, 384, 512]
  layers: [2, 3, 4, 5, 6, 8]
  heads: [2, 4, 8]
  ffn_multiplier: [2, 3, 4]
  activation: [swiglu, gelu, relu]
  normalization: [rmsnorm, layernorm]
  positional_embedding: [rope, learned, none]
  dropout: [0.0, 0.05, 0.1, 0.15, 0.2]
  seq_length: [64, 128, 256]
  vocab_size: [2000, 4000, 8000, 16000]

hyperparameter:
  learning_rate_min: 1.0e-4
  learning_rate_max: 5.0e-3
  optimizer: [adamw, sgd]
  scheduler: [cosine, linear, constant]
  warmup_ratio: [0.0, 0.05, 0.1, 0.15]
  batch_size: [2, 4, 8]
  gradient_accumulation: [4, 8, 16, 32]
  weight_decay: [0.0, 0.01, 0.1]
  gradient_clipping: [0.5, 1.0, 5.0]

hard_limits:
  max_parameters: 25000000
  max_seq_length: 256
  max_layers: 8
  max_hidden_size: 512
  max_heads: 8
  min_hidden_size: 64
  min_layers: 2
```

---

## Phase 3 — Create All Source Files

Create each file below exactly as specified.
Every file goes inside `autoslm/src/`.

---

### File: `src/logger.py`

**Purpose:** Provide a structured rotating logger used by every module.

**Must implement:**
- `get_logger(name: str, log_dir: str = "logs") -> logging.Logger`
- Logs to both stdout and `logs/autoslm.log`
- Rotating file handler: max 5 MB, 3 backups
- Format: `YYYY-MM-DD HH:MM:SS | LEVEL    | module | message`

---

### File: `src/config_loader.py`

**Purpose:** Load and merge all YAML config files into a single object used everywhere.

**Must implement:**
- `load_config(config_dir: str) -> dict` — loads and deep-merges `hardware.yaml`, `runtime.yaml`, `baseline.yaml`, `search_space.yaml` into one flat/nested dict
- `validate_config(config: dict) -> None` — raises `ValueError` if any baseline value violates a hard limit (e.g. baseline `hidden_size` > `max_hidden_size`)

---

### File: `src/hardware_monitor.py`

**Purpose:** Detect hardware and monitor resource usage during training.

**Must implement:**
- `detect_device(preferred: str) -> str` — returns `"cuda"` / `"mps"` / `"cpu"` based on availability; respects `hardware.yaml` preference
- `get_ram_used_gb() -> float` — current RAM usage in GB using `psutil`
- `is_ram_exceeded(max_gb: float) -> bool`
- `log_hardware_summary(logger)` — logs CPU count, total RAM, device, precision

---

### File: `src/tokenizer.py`

**Purpose:** Train a tokenizer on the domain dataset and save/load it.

**Must implement:**
- `train_tokenizer(data_path: str, vocab_size: int, tokenizer_type: str, save_dir: str) -> Tokenizer`
  - Reads all `content` fields from every message in the JSONL file
  - Trains BPE / WordPiece / Unigram using HuggingFace `tokenizers` library
  - Adds special tokens: `[PAD]` (id=0), `[UNK]` (id=1), `[BOS]` (id=2), `[EOS]` (id=3)
  - Saves tokenizer to `{save_dir}/tokenizer.json`
- `load_tokenizer(save_dir: str) -> Tokenizer`
- `get_vocab_size(tokenizer) -> int`

---

### File: `src/dataset.py`

**Purpose:** Load, tokenize, and batch the JSONL dataset.

**Must implement:**
- `load_jsonl(path: str) -> list[dict]`
- `AutoSLMDataset(torch.utils.data.Dataset)`
  - Each example: encode as `[BOS] + user_content + [EOS] + [BOS] + assistant_content + [EOS]`
  - Truncate to `seq_length`; labels = input_ids shifted right (standard LM objective)
  - Returns `{"input_ids": LongTensor, "attention_mask": LongTensor, "labels": LongTensor}`
- `get_dataloader(dataset, batch_size: int, shuffle: bool) -> DataLoader`
  - Custom `collate_fn` that pads each batch to the longest sequence in that batch only (dynamic padding — never pad to global max)

---

### File: `src/rope.py`

**Purpose:** Rotary positional embedding.

**Must implement:**
- `RotaryEmbedding(nn.Module)` — standard RoPE
  - `forward(seq_len: int, device) -> (cos, sin)` — returns precomputed cos/sin tensors
- `apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> (Tensor, Tensor)` — rotates Q and K in-place

---

### File: `src/attention.py`

**Purpose:** Causal multi-head self-attention with optional RoPE.

**Must implement:**
- `MultiHeadAttention(nn.Module)`
  - Constructor: `(hidden_size: int, num_heads: int, dropout: float, use_rope: bool)`
  - `forward(x: Tensor, attention_mask: Tensor = None) -> Tensor`
  - Applies RoPE to Q and K when `use_rope=True`
  - Causal mask: each token attends only to itself and previous tokens
  - Scaled dot-product attention with dropout

---

### File: `src/model.py`

**Purpose:** Full decoder-only transformer.

**Must implement:**
- `RMSNorm(nn.Module)` — root mean square normalization
- `SwiGLU(nn.Module)` — `output = x * sigmoid(gate)` where x and gate come from separate linear projections
- `TransformerBlock(nn.Module)` — one decoder layer
  - Pre-norm (norm before attention, norm before FFN)
  - MultiHeadAttention + residual
  - FFN with configurable activation + residual
- `AutoSLMModel(nn.Module)` — full model
  - Token embedding
  - N stacked TransformerBlocks
  - Final RMSNorm
  - LM head (linear, no bias, weight-tied to embedding)
  - `forward(input_ids, attention_mask=None) -> logits`  shape: `(B, T, vocab_size)`
  - `count_parameters() -> int`
- `build_model(config: dict, vocab_size: int, device: str) -> AutoSLMModel`
  - Reads: `hidden_size`, `layers`, `heads`, `ffn_multiplier`, `activation`, `normalization`, `positional_embedding`, `dropout`, `seq_length`

---

### File: `src/train.py`

**Purpose:** Training loop with early stopping and resource monitoring.

**Must implement:**
- `train(model, train_loader, val_loader, config: dict, device: str, experiment_id: str, logger) -> dict`
  - Optimizer and scheduler from config
  - Gradient accumulation: accumulate for `gradient_accumulation` micro-steps, then `optimizer.step()`
  - Gradient clipping after accumulation
  - Evaluate on `val_loader` every `eval_every_n_steps` steps (compute average val loss)
  - Check all early stopping conditions after every eval:

    | Condition | Action |
    |---|---|
    | `elapsed >= max_experiment_minutes * 60` | stop, reason = `"time_limit"` |
    | val loss has not improved for `val_stagnation_patience` evals | stop, reason = `"val_stagnation"` |
    | RAM > `max_ram_usage_gb` GB | stop, reason = `"memory_limit"` |
    | loss is NaN or Inf | stop, reason = `"divergence"` |

  - Returns `{"final_val_loss": float, "steps_completed": int, "stop_reason": str, "duration_seconds": float}`
- `build_optimizer(model, config: dict)` — supports `adamw`, `sgd`
- `build_scheduler(optimizer, config: dict, num_training_steps: int)` — supports `cosine`, `linear`, `constant`

---

### File: `src/inference.py`

**Purpose:** Batch inference and structured output parsing.

**Must implement:**
- `run_inference(model, dataset: list[dict], tokenizer, config: dict, device: str) -> list[dict]`
  - For each example: encode the user message, generate tokens greedily until `[EOS]` or `seq_length`
  - Returns list of `{"input": str, "prediction": str, "ground_truth": str}`
- `extract_matched(text: str) -> bool | None`
  - Tries to parse `overall_similarity.matched` from text as JSON
  - Returns `True`, `False`, or `None` on parse failure
- `parse_ground_truth(example: dict) -> bool | None`
  - Extracts ground truth `overall_similarity.matched` from a dataset example

---

### File: `src/evaluator.py`

**Purpose:** Compute F1, precision, recall from inference results.

**Must implement:**
- `compute_metrics(results: list[dict]) -> dict`
  - Extract predictions and ground truths using `extract_matched` and `parse_ground_truth`
  - Skip examples where either returns `None`; count them as `parse_failures`
  - Compute using `sklearn.metrics.precision_recall_fscore_support` with `average="binary"`
  - Return `{"f1": float, "precision": float, "recall": float, "support": int, "parse_failures": int}`
  - If all predictions fail to parse, return all zeros with a warning

---

### File: `src/experiment_tracker.py`

**Purpose:** Persist, query, and compare all experiment results.

**Must implement:**
- `ExperimentTracker(experiments_dir: str)`
- `log(experiment: dict) -> None` — writes `experiments/exp_NNNN.json`
- `get_best() -> dict | None` — experiment with highest `metrics.f1`
- `get_all() -> list[dict]` — all experiments sorted by timestamp ascending
- `get_last_n(n: int) -> list[dict]`
- `count() -> int`

Each experiment JSON must follow this schema exactly:

```json
{
  "experiment_id": "exp_0001",
  "timestamp": "2026-01-01T12:00:00Z",
  "parent_experiment": null,
  "mutation_type": "baseline",
  "arch_config": {},
  "train_config": {},
  "runtime": {
    "duration_seconds": 0.0,
    "stop_reason": "",
    "steps_completed": 0
  },
  "metrics": {
    "f1": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "support": 0,
    "parse_failures": 0
  },
  "result": "improved",
  "is_baseline": true
}
```

---

### File: `src/mutation_engine.py`

**Purpose:** Generate new experiment configs by mutating the best known config.

**Must implement:**
- `MutationEngine(search_space: dict, hard_limits: dict)`
- `mutate(parent_config: dict, force_strategy: str = None) -> (dict, str)`
  - Picks a mutation strategy (randomly, or forced via `force_strategy`)
  - Returns `(new_config_dict, strategy_name_used)`
  - Strategies:

    | Strategy | What changes |
    |---|---|
    | `architecture` | Randomly change 1–2 of: `hidden_size`, `layers`, `heads`, `ffn_multiplier`, `activation`, `normalization`, `positional_embedding` |
    | `hyperparameter` | Randomly change 1–2 of: `learning_rate`, `optimizer`, `scheduler`, `warmup_ratio`, `batch_size`, `gradient_accumulation` |
    | `tokenizer` | Change `vocab_size` or `tokenizer_type` |
    | `regularization` | Change `dropout`, `weight_decay`, `gradient_clipping` |
    | `random_restart` | Fully randomize all params within search space bounds |

- `is_feasible(config: dict) -> bool`
  - Estimate parameter count: `params ≈ vocab_size × hidden_size + layers × (4 × hidden_size² + ffn_multiplier × 2 × hidden_size²)`
  - Return False if estimated params > `max_parameters` or any other hard limit is violated
- `estimate_parameters(config: dict) -> int`

---

### File: `src/runtime_estimator.py`

**Purpose:** Predict whether an experiment will finish within the time budget.

**Must implement:**
- `RuntimeEstimator()`
- `estimate_minutes(config: dict, dataset_size: int) -> float`
  - Heuristic: `(dataset_size / batch_size) × seconds_per_step / 60`
  - Default `seconds_per_step = 0.5` for CPU; updated from observed history
- `update(observed_step_time: float) -> None` — exponential moving average update
- `will_exceed_budget(config: dict, dataset_size: int, budget_minutes: int) -> bool`

---

### File: `src/checkpoint_manager.py`

**Purpose:** Save and restore model + optimizer state.

**Must implement:**
- `CheckpointManager(checkpoint_dir: str)`
- `save(experiment_id: str, model, optimizer, step: int, metrics: dict) -> None`
  - Saves to `checkpoints/{experiment_id}/checkpoint.pt`
- `load(experiment_id: str, model, optimizer) -> dict` — returns `{"step": int, "metrics": dict}`
- `exists(experiment_id: str) -> bool`
- `delete(experiment_id: str) -> None` — remove checkpoint folder to reclaim disk space

---

### File: `src/search_agent.py`

**Purpose:** Decide what experiment to run next based on history.

**Must implement:**
- `SearchAgent(mutation_engine, experiment_tracker, config: dict)`
- `next_experiment() -> (dict, str, str)`
  - Returns `(config_dict, experiment_id, parent_experiment_id)`
  - Run 0: return baseline config exactly as-is, `mutation_type = "baseline"`
  - Every 10th run: force `random_restart` to escape local optima
  - Every 5th run: use whichever mutation strategy improved F1 most historically
  - All other runs: mutate the best known experiment config
- `generate_experiment_id() -> str` — returns `"exp_0001"`, `"exp_0002"`, etc. (zero-padded, auto-incrementing)
- `best_mutation_strategy(history: list[dict]) -> str` — returns the strategy with highest average F1 delta

---

### File: `src/main.py`

**Purpose:** The single entry point. Bootstraps everything. Runs the autonomous loop.

#### Startup sequence (run once on launch):

```
1. Print AutoSLM banner
2. Load config from config/ via config_loader
3. Detect and log hardware via hardware_monitor
4. Check if data/train.jsonl and data/test.jsonl exist:
     If missing → print this message and exit(0):
         "========================================"
         "  AutoSLM is waiting for your dataset."
         "  Please place train.jsonl + test.jsonl"
         "  inside the autoslm/data/ folder,"
         "  then run:  python src/main.py"
         "========================================"
5. Train tokenizer on data/train.jsonl
     → Skip if tokenizer/tokenizer.json already exists AND vocab_size matches baseline config
6. Load train dataset and test dataset
7. Create 80/20 val split from train dataset (seeded)
8. Initialize all modules:
     ExperimentTracker, MutationEngine, RuntimeEstimator,
     CheckpointManager, SearchAgent
9. Print: "Bootstrap complete. Experiments will run indefinitely. Press Ctrl+C to stop."
```

#### Autonomous loop (runs forever):

```python
while True:
    config, experiment_id, parent_id = search_agent.next_experiment()

    if not mutation_engine.is_feasible(config):
        logger.warning(f"Skipping infeasible config: {experiment_id}")
        continue

    estimated_params = mutation_engine.estimate_parameters(config)
    logger.info(f"[{experiment_id}] Starting | params={estimated_params:,} | mutation={mutation_type}")

    # Retrain tokenizer only if vocab_size or tokenizer_type changed
    if tokenizer_changed(config, last_config):
        tokenizer = train_tokenizer(...)

    model = build_model(config, tokenizer.get_vocab_size(), device)

    train_result = train(model, train_loader, val_loader, config, device, experiment_id, logger)

    runtime_estimator.update(train_result["duration_seconds"] / train_result["steps_completed"])

    results = run_inference(model, test_dataset, tokenizer, config, device)
    metrics = compute_metrics(results)

    best = tracker.get_best()
    result_label = (
        "improved" if best is None or metrics["f1"] > best["metrics"]["f1"]
        else "equal" if metrics["f1"] == best["metrics"]["f1"]
        else "regressed"
    )

    experiment_record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parent_experiment": parent_id,
        "mutation_type": mutation_type,
        "arch_config": extract_arch_config(config),
        "train_config": extract_train_config(config),
        "runtime": train_result,
        "metrics": metrics,
        "result": result_label,
        "is_baseline": mutation_type == "baseline"
    }
    tracker.log(experiment_record)

    if result_label != "improved":
        checkpoint_manager.delete(experiment_id)

    best_f1 = tracker.get_best()["metrics"]["f1"] if tracker.get_best() else 0.0
    logger.info(
        f"[{experiment_id}] F1={metrics['f1']:.4f} | "
        f"best={best_f1:.4f} | "
        f"result={result_label} | "
        f"stop={train_result['stop_reason']} | "
        f"duration={train_result['duration_seconds']:.0f}s"
    )

    last_config = config
    time.sleep(2)
```

#### On KeyboardInterrupt:

```
Print:
    "\n========================================"
    "  AutoSLM stopped by user."
    f"  Total experiments run : {tracker.count()}"
    f"  Best F1               : {best['metrics']['f1']:.4f}"
    f"  Best experiment       : {best['experiment_id']}"
    "========================================"
Exit cleanly with exit(0).
```

---

## Phase 4 — Self-Verify Before Starting the Loop

After creating all files, run these checks automatically. Fix any failure before proceeding:

| Check | Command / Action |
|---|---|
| Import check | `python -c "from src import main"` from inside `autoslm/` |
| Config check | Load all 4 YAMLs, confirm no key errors |
| Tokenizer check | Train on 10 lines of train.jsonl, save, reload, encode one sentence |
| Model check | Build baseline model, verify `count_parameters() < 10_000_000`, do one forward pass |
| Train check | Run 5 training steps, confirm loss is a valid float and decreasing |
| Inference check | Run on 3 test examples, confirm output is a string |
| Metric check | Compute F1 on dummy predictions, confirm result is float between 0 and 1 |

---

## Phase 5 — Run

```bash
cd autoslm
python src/main.py
```

---

## Constraints — Never Violate These

| Constraint | Value |
|---|---|
| Max model parameters | 25,000,000 |
| Max RAM usage | 14 GB |
| Max experiment duration | 10 minutes |
| Max sequence length | 256 tokens |
| Max layers | 8 |
| Max hidden size | 512 |
| Device requirement | Must work on CPU with no GPU |
| Internet access | None — everything runs fully offline |
| Experiment isolation | A crash in one experiment must never affect the next |

---

## Dataset Format

Each line in `train.jsonl` and `test.jsonl`:

```json
{
  "messages": [
    { "role": "user",      "content": "input text" },
    { "role": "assistant", "content": "{\"overall_similarity\": {\"matched\": true}}" }
  ]
}
```

**Primary metric:** F1-score on `overall_similarity.matched` (binary classification: `true` / `false`)

---

## Baseline Architecture (Starting Point)

| Parameter | Value |
|---|---|
| Hidden size | 256 |
| Layers | 4 |
| Attention heads | 4 |
| FFN multiplier | 4× |
| Sequence length | 128 |
| Vocab size | 8,000 |
| Activation | SwiGLU |
| Positional embedding | RoPE |
| Normalization | RMSNorm |
| Estimated parameters | ~7M |

---

## Expected Output After 50+ Experiments

```
experiments/     → 50+ JSON files, one per experiment
checkpoints/     → Best model checkpoint saved
logs/autoslm.log → Full run history
stdout           → Live summary line after every experiment
```

F1 should trend upward across the experiment history as the mutation engine discovers better configurations.

---

*Feed this file to GitHub Copilot Agent — it will build and run everything autonomously.*
