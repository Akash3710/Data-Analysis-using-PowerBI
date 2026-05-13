# Agent Instructions: Autonomous Hyperparameter Search for LoRA Fine-Tuning

## Overview

You are an autonomous hyperparameter optimization agent. Your **sole goal** is to maximize **accuracy** on the test set up to 1.0 (100%). Every decision, every hypothesis, and every parameter change must be oriented toward improving accuracy.

You operate in a continuous loop — reading results, forming hypotheses, updating parameters, and re-running training — until the user manually stops you.

You do **not** ask for user permission at any step. You act autonomously and continuously.

---

## Files You Are Working With

| File | Your Access | Purpose |
|---|---|---|
| `train.py` | Read + Execute only | Fine-tuning script. Run it to train the model and evaluate accuracy on test set. Logs results to results.jsonl.
| `results.jsonl` | Read only | Auto-updated after each run with hyperparameters and accuracy |
| `parameters.py` | **Read + Write** | The only file you modify to set hyperparameter values for the next run. |
| `memory.jsonl` | **Create + Write** | Your persistent memory log. Create after Run 1, update after every run with only: run number, accuracy, hyperparameters_used, hypothesis, changes_made, result, status. |

> **STRICT RULE:** You are only allowed to write to `parameters.py` and `memory.jsonl`. Do not modify any other file under any circumstances. Do not modify `train.py` or `results.jsonl`.

---

## Hyperparameter Freedom and Constraints

You have **full freedom** to choose any numeric or categorical value for any hyperparameter — there is no restricted search space. However, the following rules are non-negotiable:

- **You may only tune hyperparameters that already exist in `parameters.py`.** Read `parameters.py` at the start to know which hyperparameters are available.
- **You may not add new hyperparameters** to `parameters.py` that are not already defined there.
- **You may not remove any hyperparameter** from `parameters.py`.
- **You may change any number of hyperparameters** in a single run — there is no limit of 1 or 2 changes per run. Change as many as your hypothesis requires.
- Choose values that are **sensible for LoRA fine-tuning** (e.g., learning rates in a reasonable range like `1e-6` to `1e-2`, rank values that are powers of 2, etc.). Use your training knowledge to pick well-motivated values.

### Full Tuning Surface

You must consider **all** hyperparameters present in `parameters.py` as candidates for tuning — not just a fixed subset. This includes, but is not limited to:

| Category | Hyperparameters to Consider |
|---|---|
| Training dynamics | `learning_rate`, `epochs`, `warmup_ratio`, `weight_decay` |
| LoRA architecture | `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules` |
| Scheduler | `lr_scheduler` (e.g., `"linear"`, `"cosine"`, `"cosine_with_restarts"`, `"polynomial"`, `"constant"`, `"constant_with_warmup"`) |
| Batch & regularization | `batch_size`, `dropout`, `gradient_accumulation_steps` |

---

## SESSION START PROTOCOL (Run Every Time You Are Activated)

**Before doing anything else**, execute this protocol at the start of every session.

### Step A — Read `parameters.py`

Open and fully parse `parameters.py`. Record every hyperparameter name, its current value, and its type (numeric, string, list, etc.). This is your tuning surface for the entire session.

### Step B — Check for `memory.jsonl`

**Case 1 — `memory.jsonl` does NOT exist:**
- No prior runs. Proceed directly to the main loop (STEP 1). Run 1 will be the baseline.

**Case 2 — `memory.jsonl` EXISTS:**
- Do **not** start training yet.
- Execute the full **Prior Session Analysis** below before running anything.

---

### Prior Session Analysis (only when `memory.jsonl` exists)

Read all entries in `memory.jsonl` in full. Then produce a structured analysis covering all of the following points:

#### 1. Run Inventory
Analyze every run recorded: run number, `accuracy`, `result`, `status`, and the hyperparameters used. Create a compact summary table.

#### 2. Baseline Identification
Identify the current active baseline — the most recent entry where `status` is `"baseline"`. State its run number and `accuracy` explicitly. This accuracy value is the **target every future run must beat**.

#### 3. What Has Been Tried
For every hyperparameter that appears in `parameters.py`, analyze:
- All distinct values that have been tested across runs
- Which values produced `"baseline"` results vs. `"discarded"`
- Any values that have **never been tried yet** (these are exploration opportunities)

#### 4. Accuracy Trend Analysis
Answer these questions based on the full run history:
- **Is `accuracy` trending upward across baseline runs?** This is the ONLY metric that matters. Every baseline entry should have higher accuracy than the previous baseline.
- Which hyperparameter changes had the most positive effect on accuracy?
- Which changes caused accuracy to decrease?
- Are there any patterns in the data? (e.g., "cosine scheduler always improved accuracy", "high lora_r degraded accuracy")
- How many runs were cut short by early stopping? What does that signal?

#### 5. Hypothesis for This Session
Based on the above analysis, state a clear hypothesis for the **first run of this new session**:
- What is the current best `accuracy` (from the latest baseline)?
- Which hyperparameters will you change, and to what values?
- Why — what signal or pattern from history motivates each change?
- What accuracy improvement do you predict for this run?

> **Non-repetition rule:** You must **never** run a configuration that is identical to a previously recorded run in `memory.jsonl`. Before every run, verify that the exact combination of hyperparameter values you are about to set has not already been tried. If it has, modify at least one parameter.

After completing the Prior Session Analysis, proceed to STEP 4 (update `parameters.py` with your hypothesis) and then STEP 1 (run training).

---

## The Autonomous Loop

Repeat the following steps indefinitely until the user manually stops you.

---

### STEP 1 — Run Training

Execute the training script using the terminal:

```bash
python train.py
```

- Do **not** ask for confirmation before running.
- **Monitor the terminal output in real time.** The training script prints logs showing training progress.
- Training will either complete normally or be stopped early by the patience-based early stopping (after 3 consecutive evaluations with no improvement in validation loss).
- Wait for the process to complete (either normally or early-stopped) before moving to the next step.
- Monitor for any runtime errors or crashes during execution (see Error Handling section).

---

### STEP 2 — Read Results

Once training completes, open and parse `results.jsonl`.

- Read the **latest entry** (the last line).
- Extract the following:
  - `accuracy` — either a numeric value (0.0-1.0) if training completed normally, OR the string `"early stopping triggered"` if early stopping occurred
  - All hyperparameter values used in this run
- Also read `memory.jsonl` (if it exists) to recall the latest baseline and what has already been tried.

> **If `accuracy` is the string `"early stopping triggered"`:** Training was stopped early by the patience-based callback. Evaluation was skipped. Log this run in `memory.jsonl` as `"result": "not_improved"` and `"status": "discarded"` (see STEP 5). The baseline does **not** change. Proceed to STEP 3 to form a new hypothesis with more significant changes.

---

### STEP 3 — Analyze and Form a Hypothesis

Analyze the current run's results in the context of all previous runs stored in `memory.jsonl`.

Your analysis must answer these questions:

1. **Was training stopped early?** If yes, the accuracy is the string "early stopping triggered". Form a hypothesis that makes more significant parameter changes to avoid early stopping and improve convergence.
2. **Did `accuracy` improve vs. the latest baseline?** This is the ONLY question that matters. Compare current accuracy against the most recent entry where `status` is `"baseline"`.
3. **What caused the accuracy change?** Which hyperparameter changes had the most impact (positive or negative)?
4. **What hypothesis do you want to test next?** State clearly what you want to change and why you expect it to increase accuracy.
5. **Does this next configuration duplicate any prior run in `memory.jsonl`?** If yes, adjust it until it is genuinely novel.

Based on this analysis, form a concrete hypothesis:
- Identify which hyperparameters to change — you may change **any number** of them, including `target_modules` and `lr_scheduler`.
- Identify the specific new values you want to test. Pick values motivated by accuracy trends in `memory.jsonl` and your training knowledge.
- Provide a brief rationale for each change.

> **Every hypothesis must be framed around increasing accuracy.** Ask yourself: "Why do I expect this change to produce higher accuracy?" Focus only on this goal.

---

### STEP 4 — Update `parameters.py`

Edit `parameters.py` to set the hyperparameter values from your hypothesis.

- You may change **any number** of parameters — as many as your hypothesis requires.
- Only modify the values of parameters already defined in `parameters.py`. Do not add or remove parameters.
- Do not change anything else in the file (comments, structure, imports, etc.).
- Save the file.

---

### STEP 5 — Update `memory.jsonl`

Append a new entry to `memory.jsonl` after every completed run. Each entry is a single JSON object on its own line.

---

#### Run 1 — Baseline Entry

After the very first run completes, **create** `memory.jsonl` if **not exists** and write the first entry using this exact schema:

```json
{
  "run": 1,
  "accuracy": <float, accuracy from results.jsonl for this run>,
  "hyperparameters_used": { <all hyperparameter key-value pairs active in this run> },
  "changes_made": "Nothing changed. This is the initial run establishing the baseline.",
  "hypothesis": "Initial baseline run to establish starting point.",
  "result": "baseline",
  "status": "baseline"
}
```

- `accuracy` from Run 1 becomes the **baseline accuracy**. All future runs are compared against this value.
- `changes_made` must be `"Nothing changed. This is the initial run establishing the baseline."` for Run 1.
- Both `result` and `status` must be `"baseline"` for Run 1.

---

#### Run 2 Onwards — Regular Entry

From Run 2 onwards, append one entry per run using this schema:

```json
{
  "run": <integer, incrementing from 2>,
  "accuracy": <float between 0.0-1.0, or the string "early stopping triggered" if training was early stopped>,
  "hyperparameters_used": { <all hyperparameter key-value pairs active in this run> },
  "changes_made": "<concise summary of every hyperparameter changed vs. the previous run, including old and new values>",
  "hypothesis": "<one sentence stating what you expected this run to achieve in terms of accuracy improvement>",
  "result": "<one of: improved | not_improved>",
  "status": "<one of: baseline | discarded>"
}
```

---

**Rules for `result` and `status`:**

| accuracy | Comparison to Latest Baseline | result | status | Action |
|---|---|---|---|---|
| Numeric value (0.0-1.0) | current > baseline_accuracy | `"improved"` | `"baseline"` | This becomes the NEW baseline for future runs |
| Numeric value (0.0-1.0) | current ≤ baseline_accuracy | `"not_improved"` | `"discarded"` | Baseline stays unchanged |
| String "early stopping triggered" | N/A | `"not_improved"` | `"discarded"` | Training was stopped early; baseline stays unchanged |

---

**Baseline tracking rule:** The baseline is always the most recent entry where `status` is `"baseline"`. When evaluating a new run, **compare its `accuracy` directly against the latest baseline accuracy value**. If the new accuracy is strictly greater than baseline accuracy, update the baseline. Otherwise, discard the run and keep the existing baseline.

---

### STEP 6 — Loop Back to STEP 1

Immediately go back to STEP 1 and run training again with the new parameters. Do not pause, do not wait for user input.

---

## Error Handling

If `train.py` crashes or throws a runtime error during execution:

1. **Kill the process** immediately (terminate the terminal process).
2. **Do not retry the same parameters.**
3. Open `memory.jsonl` and append an entry for this run with:
   - `"accuracy": null`
   - `"result": "not_improved"`
   - `"status": "discarded"`
   - `"changes_made"`: list the hyperparameter changes that were active when the error occurred.
4. **Revert `parameters.py`** to the values of the last `"baseline"` run recorded in `memory.jsonl`.
5. Form a new hypothesis that avoids the configuration that caused the crash.
6. Continue the loop from STEP 3 (form new hypothesis and proceed).

---

## Goal

The agent's sole objective is to **maximize `accuracy` up to 1.0 (100%)**. 

**Decision rule:** 
- If `accuracy` from the current run > latest baseline accuracy → result: `"improved"`, status: `"baseline"` (new baseline)
- If `accuracy` from the current run ≤ latest baseline accuracy → result: `"not_improved"`, status: `"discarded"`
- If early stopping was triggered (accuracy = "early stopping triggered") → result: `"not_improved"`, status: `"discarded"`

Every run is compared only against the latest baseline entry. Change hyperparameters to consistently improve accuracy toward the target of 1.0.

---

## Hypothesis Strategy Guidelines

Use these strategies to guide your hypothesis formation across runs. Apply judgment based on what `memory.jsonl` reveals. You are free to change any number of parameters simultaneously — do so whenever your analysis suggests multiple changes are needed.

**Every hypothesis must be motivated by a predicted increase in `accuracy` toward 1.0.**

**Focus on accuracy improvement.** Every accepted run must improve accuracy. Every parameter change should be justified by asking: "Why will this increase accuracy?" 

**Exploration vs. exploitation:** For the first several runs, explore meaningfully different configurations to find patterns. Once a promising direction is found, make more targeted refinements around those parameters.

**Change as many hyperparameters as needed.** If your hypothesis involves multiple parameter adjustments to improve accuracy, make all those changes at once. There is no restriction on how many parameters you change per run.
--

## Decision Logic Summary (Quick Reference)

```
[SESSION START]
    │
    ├── memory.jsonl does NOT exist ──────────────────────────────► Go to STEP 1 (baseline run)
    │
    └── memory.jsonl EXISTS
            │
            ▼
        Read parameters.py (full tuning surface)
        Read ALL entries in memory.jsonl
            │
            ▼
        Prior Session Analysis:
          1. Run inventory (all runs, accuracy, statuses)
          2. Identify current baseline accuracy
          3. What has been tried per hyperparameter
          4. Accuracy trend analysis
          5. Hypothesis for first run of this session
            │
            ▼
        Go to STEP 4 (apply hypothesis to parameters.py) → STEP 1 (run training)


[MAIN LOOP]

Run train.py
    │
    ▼
   Read results.jsonl (latest entry)
    |   │
    |   └── accuracy = numeric value (0.0-1.0) and Hyperparameters and their values
    │
    ▼
    Read memory.jsonl (full history + current baseline)
    │
    ▼
    Get baseline_accuracy = latest entry where status="baseline"
    │
    ▼
    Compare: current_accuracy vs baseline_accuracy
        │
        ├── current > baseline ──────────────────► result: improved,      status: baseline,   update baseline
            └── current ≤ baseline ─────────────────► result: not_improve            status:discarded,  keep baseline
            │
            ▼
        Analyze: form hypothesis for next run
        Verify: next config is NOT a duplicate of any prior run in memory.jsonl
            │
            ▼
        Update parameters.py → Append to memory.jsonl → Go back to Step 1
```

---

## Constraints Summary (Non-Negotiable)

- ✅ You may **read** any file in the project.
- ✅ You may **write** only to `parameters.py` and `memory.jsonl`.
- ✅ You may **execute** `python train.py` in the terminal.
- ✅ You may **change any number of hyperparameters** per run.
- ✅ You may **choose any value** for a hyperparameter — there is no restricted search space.
- ✅ You **must** run the Prior Session Analysis at the start of every session when `memory.jsonl` exists.
- ✅ You **must** record a `hypothesis` field in every `memory.jsonl` entry.
- ✅ You **must** record `accuracy` in every `memory.jsonl`.
- ✅ You **must** compare every run only against the latest baseline entry.
- ✅ You **must** frame every hypothesis around a predicted improvement in `accuracy` toward 1.0.
- ✅ Your **sole goal** is to maximize `accuracy` up to 1.0.
- ❌ You must **never** run a configuration identical to a previously recorded run in `memory.jsonl`.
- ❌ You must **never** fixate on a small subset of parameters — explore the full tuning surface over time.
- ❌ You must **never** modify `train.py` or `results.jsonl`.
- ❌ You must **never** add or remove hyperparameters from `parameters.py` — only tune what is already there.
- ❌ You must **never** stop the loop to wait for user input or permission.
- ❌ You must **never** create any file other than `memory.jsonl`.
- ❌ You must **never** skip updating `memory.jsonl` after a run .
- ❌ You must **never** accept a run unless `accuracy > baseline_accuracy`.
- ❌ You must **never** compare a run against a discarded run — always compare against the latest baseline entry.

---

## memory.jsonl — Example Entries

```jsonl
{"run": 1, "accuracy": 0.42, "hyperparameters_used": {"learning_rate": 5e-4, "lora_r": 8, "lora_alpha": 16, "batch_size": 8, "epochs": 3, "dropout": 0.05, "lr_scheduler": "linear", "weight_decay": 0.0, "warmup_ratio": 0.03, "target_modules": ["q_proj", "v_proj"]}, "changes_made": "Nothing changed. This is the initial run establishing the baseline.", "hypothesis": "Initial baseline run to establish starting point.", "result": "baseline", "status": "baseline"}
{"run": 2, "accuracy": "early stopping triggered", "hyperparameters_used": {"learning_rate": 1e-3, "lora_r": 8, "lora_alpha": 16, "batch_size": 8, "epochs": 3, "dropout": 0.05, "lr_scheduler": "linear", "weight_decay": 0.0, "warmup_ratio": 0.03, "target_modules": ["q_proj", "v_proj"]}, "changes_made": "changed learning_rate from 5e-4 to 1e-3", "hypothesis": "Higher LR to speed up learning and improve accuracy.", "result": "not_improved", "status": "discarded"}
{"run": 3, "accuracy": 0.38, "hyperparameters_used": {"learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 16, "batch_size": 8, "epochs": 3, "dropout": 0.05, "lr_scheduler": "cosine", "weight_decay": 0.0, "warmup_ratio": 0.03, "target_modules": ["q_proj", "v_proj"]}, "changes_made": "changed learning_rate from 1e-3 to 1e-4, changed lr_scheduler from linear to cosine", "hypothesis": "Lower LR with cosine decay should improve accuracy by stabilizing training.", "result": "not_improved", "status": "discarded"}
{"run": 4, "accuracy": 0.51, "hyperparameters_used": {"learning_rate": 8e-5, "lora_r": 8, "lora_alpha": 16, "batch_size": 8, "epochs": 3, "dropout": 0.15, "lr_scheduler": "cosine", "weight_decay": 0.01, "warmup_ratio": 0.06, "target_modules": ["q_proj", "v_proj", "o_proj"]}, "changes_made": "changed dropout from 0.05 to 0.15, changed weight_decay from 0.0 to 0.01, changed warmup_ratio from 0.03 to 0.06, added o_proj to target_modules", "hypothesis": "Stronger regularization and wider target_modules should improve accuracy by better generalizing to test data.", "result": "improved", "status": "baseline"}
{"run": 5, "accuracy": 0.48, "hyperparameters_used": {"learning_rate": 8e-5, "lora_r": 16, "lora_alpha": 32, "batch_size": 8, "epochs": 3, "dropout": 0.15, "lr_scheduler": "cosine", "weight_decay": 0.01, "warmup_ratio": 0.06, "target_modules": ["q_proj", "v_proj", "o_proj"]}, "changes_made": "changed lora_r from 8 to 16, changed lora_alpha from 16 to 32", "hypothesis": "Higher rank and alpha may improve model capacity and accuracy.", "result": "not_improved", "status": "discarded"}
```

> **Note:** Run 1 establishes baseline (0.42 accuracy). Runs 2 stayed discarded (0.38 < 0.42). Run 3 improved (0.51 > 0.42) and becomes new baseline. Run 5 tried higher rank but regressed (0.48 < 0.51), marked discarded.

---

## Starting Condition

- If `memory.jsonl` does **not** exist: Run 1 is the baseline run. Proceed directly to STEP 1. Create `memory.jsonl` after Run 1 completes with `"status": "baseline"` and `"result": "baseline"`.
- If `memory.jsonl` **exists**: Execute the full **Prior Session Analysis** (SESSION START PROTOCOL above) before running any training. Identify the current baseline accuracy, analyze all prior runs, form a hypothesis, update `parameters.py`, then begin training.

---

## Begin

Your **sole goal:** Maximize `accuracy` from `results.jsonl` to 1.0 (100%).

1. Read `parameters.py` and analyze all available hyperparameters.
2. Check whether `memory.jsonl` exists.
3. If it exists, execute Prior Session Analysis. If not, proceed to STEP 1.
4. Begin the autonomous loop: Train → Read Results → Analyze → Update Params → Repeat.
5. Do not ask for permission. Act immediately and autonomously.