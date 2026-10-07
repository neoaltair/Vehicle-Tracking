# Kaggle M4 Frozen Research Evaluation Workflow

This document details the exact, reproducible workflow to run M4 on Kaggle GPU/CPU.
The M3 appearance-only baseline is **strictly frozen**:
- **mAP**: 26.6684%
- **Rank-1**: 40.2660%
- **Rank-5**: 54.9468%
- **Rank-10**: 62.3404%
- **GT tracklets**: 3,029
- **Evaluation queries**: 1,880

---

## 1. Setup & Environment

Pull the latest repository on Kaggle and install dependencies:

```bash
!git pull origin main
!pip install -e ".[dev]"
```

Define paths:
```bash
CITYFLOW_ROOT=/kaggle/input/<your-cityflow-dataset>/<cityflow-root>
```

---

## 2. Step 1: Validation-Only Hyperparameter Selection

Run grid search across $\alpha \in \{0.05, 0.10, 0.20\}$ and time window $T \in \{60, 120, 300, 600\}\text{s}$ **strictly using validation identities** (`splits.val`):

```bash
!python scripts/tune_m4_hyperparameters.py \
  --root "$CITYFLOW_ROOT" \
  --embedding-cache outputs/cache/embeddings \
  --min-edge-samples 5 \
  --out-dir outputs/m4_validation
```

Outputs:
- `outputs/m4_validation/val_grid_results.json`
- `outputs/m4_validation/best_config.json` (frozen parameter choice)

---

## 3. Step 2: Full M4 Ablation & Test Evaluation

Run all 4 ablations on the **test split only** using the frozen best parameters from Step 1:

```bash
!python scripts/run_m4_ablation_study.py \
  --root "$CITYFLOW_ROOT" \
  --embedding-cache outputs/cache/embeddings \
  --best-config outputs/m4_validation/best_config.json \
  --min-edge-samples 5 \
  --out-dir outputs/m4_ablation
```

This evaluates:
1. **Appearance only** (Method A)
2. **Appearance + fixed time window** (Method B)
3. **Appearance + camera topology** (Method B + topology filtering)
4. **Appearance + topology + travel-time prior** (Method C, full spatio-temporal)

Outputs:
- `outputs/m4_ablation/ablation_summary.json` (mAP, Rank-1/5/10, candidate reduction, true-match survival for each ablation)
- `outputs/m4_ablation/test_final_results.json` (Final frozen test results)
- `outputs/m4_ablation/final_test_per_query.csv`

---

## 4. Step 3: Controlled Query Degradation Study

Evaluate the robustness of Appearance-only vs. Spatio-Temporal retrieval on the test set under controlled query degradations:
- **Downscaling**: factors $\{1.0, 0.5, 0.25, 0.125\}$
- **Gaussian blur**: $\sigma \in \{0.0, 1.0, 2.0, 4.0\}$
- **Crop**: area retention $\{1.0, 0.75, 0.50, 0.25\}$
- **Occlusion**: ratios $\{0.0, 0.25, 0.50\}$ across seeds $\{0, 1, 2\}$

```bash
!python scripts/run_m4_degradation_study.py \
  --root "$CITYFLOW_ROOT" \
  --embedding-cache outputs/cache/embeddings \
  --weights outputs/weights/veri_sbs_R50-ibn.pth \
  --time-window-s 300.0 \
  --alpha 0.10 \
  --out-dir outputs/m4_degradation
```

Outputs:
- `outputs/m4_degradation/degradation_results.json` (Mean +/- std for all degradation families and severity levels)
