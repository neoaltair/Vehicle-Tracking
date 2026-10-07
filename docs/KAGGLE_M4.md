# Kaggle M4 Workflow: Spatio-Temporal Prior Evaluation

This document details running the M4 camera-transition graph and travel-time spatio-temporal prior evaluation on Kaggle GPU/CPU.

The M3 appearance-only baseline is **frozen**:
- **mAP**: 26.6684%
- **Rank-1**: 40.2660%
- **Rank-5**: 54.9468%
- **Rank-10**: 62.3404%
- **GT tracklets**: 3029
- **Evaluation queries**: 1880

---

## 1. Setup & Environment

Ensure you are in the repository root with dependencies installed:

```bash
!git pull origin main
!pip install -e ".[dev]"
```

Set the CityFlowV2 path:
```bash
CITYFLOW_ROOT=/kaggle/input/<your-cityflow-dataset>/<cityflow-root>
```

---

## 2. Temporal Synchronization Verification (Important Pre-Check)

Before relying heavily on fine-grained time bounds, inspect the camera time offsets in CityFlow:

```bash
!python scripts/inspect_cityflow.py \
  --root "$CITYFLOW_ROOT" \
  --out outputs/m4_inspect/camera_sync_summary.json
```

CityFlow scenario cameras are recorded across distinct intersections. If timestamps are frame-based without GPS absolute sync offsets, cross-scenario transitions are uninformative. Our `CameraTransitionGraph` automatically filters transitions exceeding `max_travel_time_s` (1 hour) and requires `min_edge_samples >= 2` observed on training identities.

---

## 3. Run M4 Spatio-Temporal Prior Evaluation

Once the embeddings are cached (from M3), execute the M4 evaluation:

```bash
!python scripts/run_m4_spatiotemporal.py \
  --root "$CITYFLOW_ROOT" \
  --embedding-cache outputs/cache/embeddings \
  --m3-summary outputs/m3_appearance/summary.json \
  --time-window-s 300.0 \
  --alpha 0.1 \
  --out-dir outputs/m4_spatiotemporal
```

### Outputs Produced:
- `outputs/m4_spatiotemporal/summary.json`: mAP, Rank-1, Rank-5, Rank-10, candidate reduction, and true-match survival.
- `outputs/m4_spatiotemporal/comparison.json`: Direct comparison table against frozen M3 baseline with delta metrics.
- `outputs/m4_spatiotemporal/per_query_results.csv`: Per-query retrieval ranks and AP.

---

## 4. Hyperparameter Ablation on Validation (alpha & window)

To evaluate sensitivity to the prior weight `alpha` and time window `T`:

```bash
for a in 0.05 0.1 0.2; do
  python scripts/run_m4_spatiotemporal.py \
    --root "$CITYFLOW_ROOT" \
    --embedding-cache outputs/cache/embeddings \
    --alpha $a \
    --out-dir outputs/m4_spatiotemporal/alpha_$a
done
```
