# Kaggle M3 Workflow

Heavy M3 runs should execute on Kaggle GPU. The repository remains the source of
truth; notebook cells should only orchestrate commands.

## Dataset Attachment

1. Obtain CityFlowV2 through the official AI City Challenge data access route.
2. Create a private Kaggle dataset containing the extracted CityFlowV2 folder or
   the official archives.
3. Attach that Kaggle dataset to the notebook.
4. Identify the actual dataset root, for example:

   ```text
   /kaggle/input/<your-cityflow-dataset>/AIC21_Track3_MTMC_Tracking/
   ```

Do not commit CityFlowV2, extracted frames, crops, videos, or embeddings to Git.

## Notebook Cells

```bash
!git clone https://github.com/neoaltair/Vehicle-Tracking.git
%cd Vehicle-Tracking
!pip install -e ".[dev]"
```

```bash
!python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
PY
```

Set the dataset root to the attached Kaggle path:

```bash
CITYFLOW_ROOT=/kaggle/input/<your-cityflow-dataset>/<cityflow-root>
```

Inspect structure first:

```bash
!python scripts/inspect_cityflow.py \
  --root "$CITYFLOW_ROOT" \
  --max-cameras 5 \
  --out outputs/m3_inspect/summary.json
```

Run the protocol smoke test:

```bash
!python scripts/run_m3_protocol_smoke.py \
  --root "$CITYFLOW_ROOT" \
  --max-tracklets 500 \
  --time-window-s 300 \
  --out outputs/m3_smoke/summary.json
```

Only after these pass should we run embedding extraction/evaluation over larger
subsets or the full benchmark.

Extract crops and cache embeddings for all GT tracklets:

```bash
!python scripts/extract_m3_embeddings.py \
  --root "$CITYFLOW_ROOT" \
  --crops-dir data/processed/cityflowv2/crops \
  --weights outputs/weights/veri_sbs_R50-ibn.pth \
  --embedding-cache outputs/cache/embeddings
```

Once CityFlowV2 embeddings have been cached per GT-derived tracklet, run the
appearance-only evaluator:

```bash
!python scripts/run_m3_appearance_baseline.py \
  --root "$CITYFLOW_ROOT" \
  --embedding-cache outputs/cache/embeddings \
  --out-dir outputs/m3_appearance
```

## Expected Smoke-Test Outcome

The smoke test should report:

- discovered GT-derived tracklets,
- identity split counts,
- upstream eligible query/gallery pairs,
- upstream positive pairs,
- fixed-window eligible/positive pairs.

If these counts are zero or unexpectedly small, inspect the dataset structure and
annotation files before running any heavy model inference.
