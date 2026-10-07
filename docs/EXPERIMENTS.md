# Experiments

## Research Question

> **Can appearance-based vehicle re-identification, augmented with a spatio-temporal
> transition prior derived from the road network topology, significantly improve
> cross-camera vehicle search accuracy compared to appearance alone?**

Secondary questions:
- How much does controlled query degradation affect retrieval mAP and CMC?
- How much do camera/time constraints reduce the candidate set, and how often does the true match survive pruning?

All retrieval experiments use exact NumPy cosine similarity over L2-normalized
embeddings. Approximate nearest-neighbor indexing is intentionally out of scope
so that pruning and ranking effects remain isolated experimental variables.

---

## Datasets

| Dataset | Split | Status | Notes |
|---|---|---|---|
| Demo highway video | N/A (qualitative only) | ✅ Available | Wikimedia CC BY 3.0; 300-frame MP4 |
| **CityFlowV2** | train / val / test | PENDING OWNER ACTION | Primary benchmark; see `docs/DATA.md` |

> [!IMPORTANT]
> Quantitative results below will remain **N/A** until CityFlowV2 is supplied locally or attached in Kaggle.
> See `docs/DATA.md` for download instructions.

---

## Evaluation Protocol

### Metrics
| Metric | Definition |
|---|---|
| **mAP** | Mean Average Precision over all queries |
| **CMC R1** | Cumulative Match Characteristic at rank 1 (% queries where correct match is #1) |
| **CMC R5** | CMC at rank 5 |
| **Search-space reduction** | `1 - candidates_after / candidates_before`, averaged over queries |
| **True-match survival** | Fraction of queries where at least one valid positive remains after pruning |

The metric implementation must support multiple positives per query and the
project's upstream cross-camera protocol. If an external metric helper does not
match this protocol exactly, use a small NumPy implementation with a
hand-checkable unit test.

### Positive and Gallery Definition

A valid positive is a gallery observation/tracklet that satisfies all of:

- same ground-truth vehicle identity as the query,
- different camera from the query,
- earlier than the query,
- within the active method's eligible temporal/search constraints,
- otherwise eligible for the gallery split.

Multiple positives are allowed. Same-camera matches are excluded from the
controlled cross-camera evaluation. At test-time inference, ground-truth identity
is used only for evaluation, never as a retrieval input.

All other eligible gallery observations/tracklets are negatives.

### Splits and Leakage Controls

Train, validation, and test splits must be identity-disjoint. Hyperparameters
such as the search window `T`, prior weight `alpha`, thresholds, and any graph
construction choices are selected on train/validation only. Final test metrics
are reported once those choices are fixed.

Camera transition graphs and travel-time priors must be estimated from training
identities only. Validation/test identities must not influence graph edges,
travel-time statistics, or prior normalization.

For the controlled research experiment, use dataset ground-truth-derived
observations/tracklets when the dataset supports them. Keep the existing
YOLO/ByteTrack tracklets as the end-to-end demonstration pipeline.

### Methods

| Method ID | Name | Description |
|---|---|---|
| **A** | Appearance-only | Single-frame query; rank candidates by exact cosine similarity without temporal pruning |
| **B** | Appearance + upstream time window | Method A filtered to other cameras in the upstream interval `[Tq - T, Tq]`, with `T` selected on validation |
| **C** | Appearance + camera graph + travel-time prior | Method B restricted/scored with a training-only camera graph and travel-time prior; final score is `appearance_score + alpha * normalized_prior` |
| **D** | C + tracklet aggregation | Same as C, but use tracklet-level aggregation for query and gallery embeddings |

Methods A-D are implemented in order across M3-M5. M2 remains a qualitative
single-camera retrieval sanity check only.

### Degradation Ablations

Apply degradation only to the query image/tracklet. The gallery remains clean.
Use deterministic seeds `0`, `1`, and `2` for stochastic transformations and
report mean +/- standard deviation.

| Family | Levels |
|---|---|
| **Downscale + upsample** | `1`, `0.5`, `0.25`, `0.125` |
| **Gaussian blur** | Configurable sigma levels |
| **Crop** | Keep `100%`, `75%`, `50%`, `25%` |
| **Occlusion** | Cover `0%`, `25%`, `50%` |

---

## M2 Baseline Results (Demo Video — Qualitative Only)

> [!NOTE]
> The demo video contains only 6 tracklets from a single camera; there is no
> ground-truth identity annotation.  These results are **qualitative only** and
> cannot be used to compute mAP or CMC.

| Query Tracklet | Frames | Gallery Size | Top-1 Match | Top-1 Sim | Top-5 Sims |
|---|---|---|---|---|---|
| `c001_trk0012` | 96 | 5 | `c001_trk0002` | 0.9860 | 0.9860, 0.9858, 0.9823, 0.9741, 0.9723 |

**Observation:** All pairwise cosine similarities are high (> 0.97), which is expected for
vehicles on the same highway in similar lighting.  Discriminative power between different
vehicle identities cannot be assessed without multi-camera ground-truth data (CityFlowV2).

Visual result: `docs/assets/retrieval_demo_top5.png`

---

## M3 Frozen Baseline Results (CityFlowV2 Ground-Truth Tracklets)

The appearance-only baseline (Method A) was evaluated on 3,029 GT tracklets across 1,880 evaluation queries using FastReID SBS(R50-ibn) embeddings:

| Metric | M3 Value | Description |
|---|---|---|
| **mAP** | **26.6684%** | Mean Average Precision across all queries |
| **Rank-1** | **40.2660%** | Top-1 retrieval accuracy |
| **Rank-5** | **54.9468%** | Top-5 retrieval accuracy |
| **Rank-10** | **62.3404%** | Top-10 retrieval accuracy |
| **Queries** | 1,880 | Valid cross-camera upstream queries |
| **GT Tracklets** | 3,029 | Full evaluation pool |

This baseline serves as the frozen reference point for all subsequent spatio-temporal and degradation comparisons.

---

## M4 Spatio-Temporal Prior Protocol

1. **Camera Graph Estimation:** Directed graph constructed strictly from the training split identities (`splits.train`, 60% of identities).
2. **Travel-Time Models:** Empirical log-normal distributions fitted for edges with $\ge 2$ observed transitions.
3. **Candidate Filtering & Ranking:** Upstream cross-camera candidates are scored using:
   $$\text{Score} = \text{CosineSim}(q, c) + \alpha \cdot \frac{P(u \to v, \Delta t)}{\max P}$$
4. **Metrics Tracked:** mAP, Rank-1, Rank-5, Rank-10, candidate search-space reduction (`pruning_reduction`), and true-match survival (`true_match_survival`).

---

## Planned Experiment Schedule

| Milestone | Experiment | Status |
|---|---|---|
| M2 | Qualitative retrieval demo | Complete |
| M3 | Appearance-only baseline on CityFlowV2 | **Complete (Frozen: mAP 26.67%, R1 40.27%)** |
| M4 | Training-only camera graph & travel-time prior | **Implemented & Tested** |
| M5 | Multi-stage spatio-temporal search & tracklet aggregation | Planned |
| M6 | Final degradation study: all methods x all degradation families | Planned |
