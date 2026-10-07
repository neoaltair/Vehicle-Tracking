# Dataset Catalog & Data Documentation

This document records the official sources, licenses/terms, access procedures, local paths, and operational status for all datasets used across the project milestones, strictly adhering to the project data rules.

---

## 1. Demo Surveillance Video (Used in M1–M2)

- **Role:** Prototype demonstration, single-camera detection, tracking validation, and query crop extraction.
- **Source:** Wikimedia Commons
- **Official URL:** [https://commons.wikimedia.org/wiki/File:Avtocesta.webm](https://commons.wikimedia.org/wiki/File:Avtocesta.webm)
- **License / Terms:** Creative Commons Attribution 3.0 Unported (`CC BY 3.0`)
- **Direct Download URL:** `https://upload.wikimedia.org/wikipedia/commons/6/68/Avtocesta.webm`
- **Local Path:** `data/raw/samples/demo_traffic.mp4` (converted / formatted via OpenCV)
- **Status:** **ACTIVE / VERIFIED**
- **Properties Verified:**
  - Resolution: 1920 × 1080 (1080p)
  - Framerate: 30.0 FPS
  - Content: Fixed-view elevated highway CCTV observation of vehicle traffic flow with multi-lane vehicle transitions.

---

## 2. CityFlowV2 / AI City Challenge Multi-Camera Tracking Dataset (M3–M6)

- **Role:** Primary research benchmark for multi-camera vehicle search, query/gallery construction, spatio-temporal camera transition graph estimation, candidate pruning evaluation, and test benchmarks.
- **Source:** NVIDIA AI City Challenge / University of Washington (CityFlow Benchmark)
- **Official URL:** [https://www.aicitychallenge.org/2021-data-and-evaluation/](https://www.aicitychallenge.org/2021-data-and-evaluation/)
- **License / Terms:** AI City Challenge Academic Research License Agreement (Non-commercial research use only; redistribution prohibited).
- **Access Procedure:**
  1. Use the official AI City Challenge data access/download path for 2021 Track 3 / CityFlowV2.
  2. Accept the AI City Challenge Data License Agreement.
  3. Download the Track 3 multi-camera tracking dataset archives.
  4. Attach the extracted dataset to Kaggle as a private dataset, or upload the official archives and extract them in the Kaggle working directory.
- **Expected Local Structure:**
  ```
  data/raw/cityflowv2/
  ├── train/
  │   ├── S01/
  │   │   ├── c001/
  │   │   │   ├── vdo.avi
  │   │   │   ├── gt/gt.txt
  │   │   │   └── calibration.txt
  │   │   ├── c002/
  │   │   └── ...
  │   ├── S03/
  │   └── S04/
  ```
- **Status:** **ACTIVE IN BENCHMARK PIPELINE**
  > [!NOTE]
  > **Camera Timestamp Synchronization & Offset Analysis:**
  > CityFlow cameras are grouped under distinct scenarios (e.g. `S01`, `S03`, `S04`). Within each scenario, cameras operate at an estimated nominal framerate (e.g., 10.0 FPS). While cameras within a single intersection/corridor share a synchronized recording origin, cross-scenario or uncalibrated streams can exhibit drift or independent start clocks.
  > In our design:
  > 1. Timestamps default to `(frame_id - 1) / fps + time_offset_s`.
  > 2. The `CameraTransitionGraph` isolates transitions strictly between cameras sharing observed valid transitions on training identities within `max_travel_time_s` (<= 3600s), filtering uncoupled cross-scenario timing noise.
  > 3. If exact GPS synchronization offsets are absent, soft travel-time log-normal density re-ranking is preferred over hard threshold truncation.

---

## 3. Secondary Datasets (Later, Not M3 Blockers)

- **RoundaboutHD:** External real-world validation after the CityFlowV2 pipeline is complete.
- **VRIC:** Possible later appearance/Re-ID robustness benchmark.
- **SimVeRi:** Possible controlled synthetic supplementary experiment.
- **ACCIDENT/CADP:** Later incident-detection/application-layer datasets only.

These datasets do not replace CityFlowV2 and should not block M3.

---

## 4. Pretrained Vehicle Re-ID Weights (M2–M6)

- **Model:** FastReID `SBS(R50-ibn)` trained on VeRi-776
- **Architecture:** ResNet-50 with Instance-Batch Normalization (IBN) + Strong Baseline (SBS)
- **Official Source URL:** [https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth)
- **License:** Apache License 2.0 (FastReID open-source release)
- **Local Cache Path:** `outputs/weights/veri_sbs_R50-ibn.pth` (gitignored, downloaded automatically with checksum verification)
- **Performance:** 97.0% Rank-1, 81.9% mAP on VeRi-776 benchmark
- **Status:** **ACTIVE / VERIFIED** (HTTP 200, 198,261,759 bytes verified)

The project uses these weights as a pretrained vehicle Re-ID model only. VeRi-776
is not used as a project dataset or requested from the owner.
