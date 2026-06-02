# README Report Update Design

## Goal

Update the repository report so the `README.md` reflects the latest experiment outputs from the notebook and JSON logs, then publish the combined documentation and code changes to the current Git branch.

## Scope

- Use the current experiment artifacts as the source of truth:
  - `final-version.ipynb`
  - `log/step2_metrics.json`
  - `log/pytorch_metrics.json`
  - `log/step3_deepspeed_metrics.json`
- Regenerate charts in `charts/` from the current logs.
- Rewrite the results/comparison sections in `README.md` so tables and narrative match the current metrics.
- Fix stale filename references in `README.md` so run instructions match the actual repo files.
- Include the existing tracked code changes in `src/*.py` and `plot_charts.py` in the final commit/push, since the user explicitly approved pushing both the report work and the code work together.

## Data Mapping

- `step2_metrics.json`: 1 GPU baseline with gradient checkpointing.
- `pytorch_metrics.json`: 2 GPU PyTorch native pipeline runs with chunks `2, 4, 8, 16`.
- `step3_deepspeed_metrics.json`: 2 GPU DeepSpeed pipeline runs with chunks `4, 8, 16`.
- Notebook `final-version.ipynb`: supporting provenance for the experiment flow and script lineage.

## Deliverables

### README

- Add a concise experiment summary near the top.
- Replace stale metric tables with updated values from the log files.
- Add a cross-method comparison table for:
  - throughput
  - sec/step
  - peak VRAM
  - final loss
  - bubble ratio where applicable
- Keep chart references in the README and point them at regenerated images under `charts/`.
- Correct script names in the run section to the current repo layout.

### Charts

- Ensure the existing chart generator produces images from the latest logs.
- Regenerate the chart set under `charts/`.
- Preserve a report-friendly set of charts rather than embedding large raw notebook output.

### Git

- Review the full worktree.
- Stage the approved report files and the currently modified code files together.
- Commit with a message that covers documentation refresh and experiment/code sync.
- Push to the current branch.

## Approach Options

### Option 1

Only update the Markdown tables and prose in `README.md`.

Tradeoff: fastest, but leaves deleted/outdated charts unresolved.

### Option 2

Regenerate charts from current logs, refresh the README tables/narrative, and push everything together.

Tradeoff: slightly more work, but keeps the repo internally consistent and is the recommended approach.

### Option 3

Export notebook-native visuals into the README directly.

Tradeoff: heavier repo footprint and harder maintenance without meaningful reporting benefit.

## Recommended Design

Use Option 2. The repo already has a chart workflow and structured JSON logs, so the cleanest path is to regenerate the charts from current metrics, update the README to reference those exact outputs, and push the code plus documentation changes together in one branch update.

## Validation

- Verify chart generation completes successfully.
- Verify the README values match the JSON logs.
- Verify modified Python files still compile.
- Review `git diff` before commit.
- Push only after a successful local verification pass.
