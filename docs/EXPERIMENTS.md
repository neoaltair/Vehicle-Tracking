# Experiments

## Research Question

> **Can appearance-based vehicle re-identification, augmented with a spatio-temporal
> transition prior derived from the road network topology, significantly improve
> cross-camera vehicle search accuracy compared to appearance alone?**

Secondary questions:
- How much does image degradation (compression artefacts, illumination changes, occlusion) affect retrieval mAP?
- At what gallery size does the naïve cosine-similarity approach become the bottleneck, and does FAISS indexing recover throughput without accuracy loss?

---

## Datasets

| Dataset | Split | Status | Notes |
|---|---|---|---|
| Demo highway video | N/A (qualitative only) | ✅ Available | Wikimedia CC BY 3.0; 300-frame MP4 |
| **CityFlow v2** | train / val / test | ⚠️ PENDING OWNER ACTION | Requires CVPR 2021 challenge registration; see `docs/DATA.md` |
| **VeRi-776** | train / query / gallery | ⚠️ PENDING OWNER ACTION | Requires NIST data agreement; see `docs/DATA.md` |

> [!IMPORTANT]
> Quantitative results below will remain **N/A** until the owner acquires CityFlow v2 and VeRi-776 access.
> See `docs/DATA.md` for download instructions.

---

## Evaluation Protocol

### Metrics
| Metric | Definition |
|---|---|
| **mAP** | Mean Average Precision over all queries |
| **CMC R1** | Cumulative Match Characteristic at rank 1 (% queries where correct match is #1) |
| **CMC R5** | CMC at rank 5 |
| **Retrieval latency** | Wall-clock time per query (ms), measured on CPU and (future) GPU |

### Methods

| Method ID | Name | Description |
|---|---|---|
| **A** | Appearance-only | Cosine similarity on mean-pooled FastReID SBS R50-ibn embeddings |
| **B** | Appearance + time window | Method A filtered to candidates in a ±Δt temporal window (Δt tuned on val) |
| **C** | Appearance + graph prior | Method A re-ranked with spatio-temporal transition probability prior derived from road network graph |
| **D** | Appearance + window + graph | Combined B + C (full system) |

Methods A–D are implemented in order across milestones M2–M4.

### Degradation Ablations

Each method above will be evaluated under four input conditions:

| Condition | Description |
|---|---|
| **Clean** | Original frames, no augmentation |
| **Compress** | JPEG quality 20 (simulates low-bandwidth transmission) |
| **Illum** | Random brightness ±40%, contrast ±30% (simulates lighting changes) |
| **Occlude** | Random 30×30 px black patch on query crop (simulates partial occlusion) |

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
vehicle identities cannot be assessed without multi-camera ground-truth data (CityFlow v2).

Visual result: `docs/assets/retrieval_demo_top5.png`

---

## Planned Experiment Schedule

| Milestone | Experiment | Dataset Required |
|---|---|---|
| M2 | Qualitative retrieval demo (complete) | Demo video |
| M3 | Method A quantitative: mAP, CMC R1/5 on VeRi-776 | VeRi-776 |
| M3 | Degradation ablation — Method A | VeRi-776 |
| M4 | Method B quantitative: time-window filtering | CityFlow v2 |
| M4 | Method C quantitative: graph prior | CityFlow v2 |
| M4 | Method D (full system) vs. A/B/C | CityFlow v2 |
| M4 | Latency benchmarks: CPU vs GPU | Either |
| M5 | Final ablation table: all methods × all degradations | Both |
