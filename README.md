# Incident-Triggered Multi-Camera Vehicle Search

An end-to-end research prototype of an **incident-triggered multi-camera vehicle search system**.

## Scenario & Goal

When a vehicle is involved in an incident (e.g., hit-and-run, collision) but the local camera view is blurry, distant, or occluded, an operator needs to review footage from nearby connected cameras in the time window immediately preceding the incident to identify clearer views of the vehicle.

This system automates that upstream search:
1. Takes a low-quality vehicle query observation at camera $c$ and time $t$.
2. Prunes candidate cameras and time windows using a **camera graph with empirical travel-time priors**.
3. Ranks surviving vehicle tracklets using **re-identification (re-ID) embeddings**.
4. Presents a ranked list with video clips/thumbnails for human confirmation.

---



## Repository Structure

```
.
├── README.md                 # Project overview, environment, and run guide
├── pyproject.toml            # Package configuration and dependencies
├── .gitignore
├── configs/                  # Experiment and pipeline configuration YAMLs
├── docs/
│   ├── DATA.md               # Dataset catalog, licenses, local paths, and access instructions
│   ├── ARCHITECTURE.md       # Architectural diagram, schemas, and decisions log
│   ├── EXPERIMENTS.md        # Protocol, splits, seeds, commands, and results
│   ├── KAGGLE_M3.md          # Kaggle GPU workflow for CityFlowV2 smoke tests
│   ├── LIMITATIONS_ETHICS.md # Limitations, ethics, and human-in-the-loop policies
│   └── assets/               # Output figures, thumbnails, and sample frames
├── src/incident_search/
│   ├── __init__.py
│   ├── io/                   # Pydantic schemas, video utilities, dataset loaders
│   ├── detect_track/         # YOLO11 + ByteTrack detection & tracking
│   ├── reid/                 # FastReID feature extraction & tracklet aggregation
│   ├── graph/                # NetworkX camera transition graph & log-normal priors
│   ├── search/               # Spatio-temporal pruning & cosine similarity ranking
│   ├── incident/             # Rule-based kinematic incident detector
│   ├── degrade/              # Deterministic query degradation transformations
│   ├── eval/                 # Retrieval metrics (mAP, CMC) & evaluation runners
│   └── app/                  # Streamlit visual dashboard
├── scripts/                  # CLI tools for running pipeline stages
├── tests/                    # Pytest test suite
├── data/                     # Local datasets and raw samples (gitignored)
└── outputs/                  # Logs, run artifacts, and embedding cache (gitignored)
```

---

## Quickstart

### 1. Installation

```powershell
pip install -e .
pip install -e ".[dev]"
```

### 2. Run Tests

```powershell
pytest
ruff check .
```

### 3. Run Retrieval Demo (M2)

```powershell
python scripts/retrieval_demo.py
# Output: docs/assets/retrieval_demo_top5.png
```

### 4. Inspect CityFlowV2 (M3, Dataset Required)

```powershell
python scripts/inspect_cityflow.py --root data/raw/cityflowv2 --max-cameras 5
python scripts/run_m3_protocol_smoke.py --root data/raw/cityflowv2 --max-tracklets 500
python scripts/run_m3_appearance_baseline.py --root data/raw/cityflowv2 --embedding-cache outputs/cache/embeddings
```

For Kaggle GPU execution, see [docs/KAGGLE_M3.md](docs/KAGGLE_M3.md).

---

## Milestone Progress

- [x] **M0: Setup** — Packaging, environment specification, test harness, git repo.

- [x] **M1: Detection and Tracking** — YOLO11m + ByteTrack, tracklet schemas, crops, and annotated video rendering.
  - Verified on highway CCTV footage (`data/raw/samples/demo_traffic.mp4`).
  - Saved 6 persistent tracklets and 48 quality-sampled vehicle crops.
  - Sample outputs: [Sample Annotated Frame](docs/assets/sample_annotated_frame.jpg) and vehicle crops in `docs/assets/`.
- [x] **M2: Embeddings and Retrieval Demo** — FastReID SBS R50-ibn (VeRi-776) feature extractor, L2-normalized 2048-d embeddings, mean-pool tracklet aggregation, cosine-similarity retrieval, top-5 grid demo.
  - Embedded all 6 demo tracklets (48 crops); per-tracklet `.npz` cache under `outputs/cache/embeddings/`.
  - Retrieval demo output: [retrieval_demo_top5.png](docs/assets/retrieval_demo_top5.png) — query `c001_trk0012` (96 frames) vs. gallery of 5; top-1 sim = 0.9860.
  - Docs added: [ARCHITECTURE.md](docs/ARCHITECTURE.md), [EXPERIMENTS.md](docs/EXPERIMENTS.md).
- [x] **M3: CityFlowV2 Dataset and Retrieval Baseline** — GT tracklet loader, identity-disjoint splits, upstream cross-camera protocol, mAP/CMC evaluation, embedding extraction, Kaggle run guide.
  - `src/incident_search/io/cityflow.py` — discovers CityFlow cameras, parses `gt.txt`, builds GT-derived `Tracklet` objects, extracts crops from video.
  - `src/incident_search/eval/` — `protocol.py` (upstream eligible/positive masks, identity splits), `metrics.py` (mAP, CMC Rank-1/5/10, multi-positive), `baseline.py` (cosine score matrix, CSV writer).
  - Scripts: `inspect_cityflow.py`, `run_m3_protocol_smoke.py`, `extract_m3_embeddings.py`, `run_m3_appearance_baseline.py`.
  - Docs: `docs/KAGGLE_M3.md`, `docs/PRE_M3_AUDIT.md`, `docs/LIMITATIONS_ETHICS.md`.
  - Tests: 21/21 pass including `test_cityflow_loader.py` (4 tests) and `test_eval_protocol.py` (5 tests).
  - ⚠️ **Quantitative baseline pending** — requires CityFlowV2 data access (see `docs/DATA.md`). Run on Kaggle GPU following `docs/KAGGLE_M3.md`.

- [x] **M4: Camera Graph & Spatio-Temporal Prior (Frozen)** — Directed camera transition graph with log-normal travel time priors, leakage prevention, validation-only parameter tuning, ablation study, and controlled degradation evaluation.
  - `src/incident_search/graph/camera_graph.py` — `CameraTransitionGraph` and `EdgeTravelTimeModel` fitted strictly on training identities (`splits.train`) with minimum 5 edge transitions.
  - `src/incident_search/search/spatiotemporal.py` — Prior matrix computation, topology & connectivity masking, and fused scoring ($\text{score} = \text{appearance} + \alpha \cdot \text{prior}$).
  - `src/incident_search/degrade/transforms.py` — Deterministic query image degradations (downscaling, Gaussian blur, central cropping, occlusion).
  - Scripts:
    - `scripts/tune_m4_hyperparameters.py` (validation grid search for $\alpha \in \{0.05, 0.10, 0.20\}$ and $T \in \{60, 120, 300, 600\}\text{s}$).
    - `scripts/run_m4_ablation_study.py` (4 ablations: appearance-only, +time window, +topology, +travel-time prior on frozen test set).
    - `scripts/run_m4_degradation_study.py` (degraded query evaluation).
    - `docs/KAGGLE_M4.md` (reproducible Kaggle execution pipeline).
  - Tests: 29/29 pass including tests for identity-disjoint isolation, temporal filtering, survival metric, validation tuning, and degradation transforms.
  - Frozen M3 Baseline preserved: mAP 26.6684%, Rank-1 40.2660%, Rank-5 54.9468%, Rank-10 62.3404%.

- [ ] **M5: Spatio-Temporal Search** — Candidate pruning and ranked retrieval evaluation.
- [ ] **M6: Degradation Protocol** — Controlled quality degradation experiments (blur, downscale, occlusion).
- [ ] **M7: Incident Detection** — Kinematic rule-based detector connected to search trigger.
- [ ] **M8: Interactive Dashboard** — Streamlit visual investigation dashboard.
- [ ] **M9: Finalize** — Ethical review, paper notes, reproducibility checklist.
