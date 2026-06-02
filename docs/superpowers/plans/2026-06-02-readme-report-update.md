# README Report Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the experiment report so `README.md` and `charts/` match the current notebook/log outputs, then publish the approved code and documentation changes to the current Git branch.

**Architecture:** Treat the JSON logs as the single source of truth for metrics, use `plot_charts.py` to regenerate report images from those logs, and rewrite the README result sections so every table and claim matches the generated data. Include the currently modified code files in the same commit because the user explicitly approved pushing both report and code changes together.

**Tech Stack:** Python, JSON, Matplotlib, Markdown, Git

---

### Task 1: Align Metric Sources and Chart Generation

**Files:**
- Modify: `plot_charts.py`
- Read: `log/step2_metrics.json`
- Read: `log/pytorch_metrics.json`
- Read: `log/step3_deepspeed_metrics.json`

- [ ] **Step 1: Point the chart script at the current DeepSpeed log**

```python
LOG_PYTORCH = "log/pytorch_metrics.json"
LOG_DEEPSPEED = "log/step3_deepspeed_metrics.json"
LOG_1GPU = "log/step2_metrics.json"
```

- [ ] **Step 2: Remove fallback-only assumptions that would hide stale data**

```python
if os.path.exists(LOG_DEEPSPEED):
    with open(LOG_DEEPSPEED, "r") as f:
        ds_data = json.load(f)
else:
    raise FileNotFoundError(f"Missing required log: {LOG_DEEPSPEED}")
```

- [ ] **Step 3: Regenerate the chart set**

Run: `python3 plot_charts.py`

Expected: chart images recreated under `charts/` with no traceback.

### Task 2: Refresh README Metrics and Narrative

**Files:**
- Modify: `README.md`
- Read: `final-version.ipynb`
- Read: `log/step2_metrics.json`
- Read: `log/pytorch_metrics.json`
- Read: `log/step3_deepspeed_metrics.json`

- [ ] **Step 1: Replace stale script references with current repo paths**

```md
- Bước 1: `src/baseline_1gpu.py`
- Bước 2: `src/gradient_checkpointing_1gpu.py`
- Bước 3: `src/pytorch_2gpu.py`
- Bước 4: `src/deepspeed_2gpu.py`
```

- [ ] **Step 2: Add a cross-method summary table near the results section**

```md
| Phương án | Cấu hình nổi bật | Throughput (tok/s) | Sec/step | Peak VRAM |
| :-- | :-- | --: | --: | --: |
| 1 GPU + GC | batch 3, grad accum 2 | ... | ... | ... |
| PyTorch PP tốt nhất | chunks 16 | ... | ... | ... |
| DeepSpeed PP tốt nhất | chunks 16 | ... | ... | ... |
```

- [ ] **Step 3: Rewrite the detailed PyTorch and DeepSpeed result tables from the JSON logs**

```md
| Chunks | Bubble | Tokens/sec | Sec/step | Peak VRAM | Final loss |
| :--: | --: | --: | --: | --: | --: |
| 2 | 33.33% | 324.8 | 25.224 | 6.182 | 5.1055 |
```

- [ ] **Step 4: Replace stale narrative claims with short evidence-backed findings**

```md
- PyTorch native pipeline tăng throughput từ 324.8 lên 571.4 tok/s khi tăng chunks từ 2 lên 16.
- DeepSpeed pipeline cải thiện từ 269.4 lên 304.4 tok/s khi giảm bubble từ 20.00% xuống 5.88%.
- 1 GPU + gradient checkpointing vẫn là mốc tiết kiệm phần cứng, nhưng throughput thấp hơn cấu hình pipeline tốt nhất.
```

### Task 3: Verify, Commit, and Push

**Files:**
- Modify: `README.md`
- Modify: `plot_charts.py`
- Modify: `src/baseline_1gpu.py`
- Modify: `src/gradient_checkpointing_1gpu.py`
- Modify: `src/pytorch_2gpu.py`
- Modify: `src/deepspeed_2gpu.py`
- Modify/Create: `charts/*`
- Modify/Create: `log/*`

- [ ] **Step 1: Verify Python files still compile**

Run: `python -m py_compile src/deepspeed_2gpu.py src/pytorch_2gpu.py src/baseline_1gpu.py src/gradient_checkpointing_1gpu.py plot_charts.py`

Expected: no output, exit code `0`.

- [ ] **Step 2: Verify chart generation and README diff**

Run: `python3 plot_charts.py && git diff -- README.md plot_charts.py charts`

Expected: regenerated charts and a README diff that matches current logs.

- [ ] **Step 3: Review full worktree before commit**

Run: `git status --short --branch && git diff --stat`

Expected: staged scope includes the approved code changes plus report artifacts.

- [ ] **Step 4: Commit the approved combined update**

```bash
git add README.md plot_charts.py src/baseline_1gpu.py src/gradient_checkpointing_1gpu.py src/pytorch_2gpu.py src/deepspeed_2gpu.py log charts docs/superpowers/specs/2026-06-02-readme-report-update-design.md docs/superpowers/plans/2026-06-02-readme-report-update.md
git commit -m "docs: refresh experiment report and sync training artifacts"
```

- [ ] **Step 5: Push the branch**

Run: `git push origin main`

Expected: push completes successfully against the current branch tip.
