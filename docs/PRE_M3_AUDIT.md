# Pre-M3 Audit

Date: 2026-10-06

## Must Fix

### Experiment protocol drift

- **Problem:** `docs/EXPERIMENTS.md` described Method B with a symmetric
  `+/-` time window, Method D as window plus graph, and degradation using
  compression/illumination settings.
- **Evidence:** The handoff protocol requires upstream search `[Tq - T, Tq]`,
  Method D as `C + tracklet aggregation`, and degradation families downscale,
  blur, crop, and occlusion.
- **Why it matters:** These definitions control the research comparison. If
  M3-M6 were implemented from the stale text, the reported results would not
  answer the agreed research question.
- **Fix:** Updated `docs/EXPERIMENTS.md` with the agreed A-D methods, upstream
  temporal rule, degradation families, search-space reduction, and true-match
  survival metrics.

### Positive/gallery definition missing

- **Problem:** The repository did not explicitly define positives, negatives,
  same-camera exclusions, multiple positives, or upstream eligibility.
- **Evidence:** The schema supports `gt_vehicle_id`, but the experiment doc did
  not state how it is used for evaluation.
- **Why it matters:** mAP/CMC can change materially depending on gallery
  filtering. M3 cannot produce valid baseline numbers without this definition.
- **Fix:** Added the positive/gallery definition to `docs/EXPERIMENTS.md`.

### Leakage controls needed before implementation

- **Problem:** The repository did not explicitly state identity-disjoint splits
  or training-only graph/travel-time construction in the experiment protocol.
- **Evidence:** CityFlowV2 access is pending, and graph/search modules are not
  implemented yet.
- **Why it matters:** Leakage through split construction or camera graph
  statistics would invalidate validation/test results.
- **Fix:** Added split, hyperparameter-selection, and graph-prior leakage rules
  to `docs/EXPERIMENTS.md`.

## Already Adequate

- **M0 package/test harness:** Checked with `python -m pytest -q` and
  `python -m ruff check .`; all tests and lint checks pass. No change required.
- **M1 detection/tracking path:** YOLO11m + Ultralytics ByteTrack are implemented
  in `src/incident_search/detect_track/tracker.py`; outputs exist under
  `outputs/m1_run/`. No change required.
- **M2 embedding/retrieval demo path:** FastReID SBS R50-ibn extraction,
  L2-normalization, `single`/`mean` aggregation, `.npz` cache, and exact cosine
  demo ranking are implemented. No change required.
- **Dataset access rules:** `docs/DATA.md` correctly marks CityFlowV2 as pending
  owner action and avoids unofficial mirrors. No change required.
- **Existing code surface:** There are no implemented graph/search/degradation
  modules yet, so no runtime graph leakage or incorrect M5 pruning code was found.
  No change required.

## Optional Later

- Add lightweight validators for confidence ranges, box coordinate validity, and
  aligned list lengths in `Tracklet`.
- Add model/cache metadata beyond the current model subdirectory if multiple
  checkpoints with the same model name will be used.
- Split model-loading integration tests from fast unit tests if CPU test time
  becomes painful.
- Add `docs/LIMITATIONS_ETHICS.md` before finalization, or remove the README
  tree entry until the file exists.

## Changes Made

- Corrected the A-D method definitions, upstream window, degradation protocol,
  metrics, positive/gallery definition, and leakage controls in
  `docs/EXPERIMENTS.md`.
- Corrected the architecture decision log to keep approximate indexing out of
  scope for the current research protocol.
- Corrected the embedding path description from `.npy` to `.npz` in
  `src/incident_search/io/schema.py`.
- Corrected the default re-ID weights path in `configs/default.yaml` to match
  the actual local/cache convention.

## Current Status

M3 is ready to begin.

The next implementation step is to supply CityFlowV2 through the official AI City
Challenge access path, then build dataset loaders, identity-disjoint split files,
controlled GT-derived observations/tracklets, and a metric implementation that
matches the multi-positive upstream cross-camera protocol.
