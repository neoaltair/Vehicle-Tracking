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

## 2. CityFlow / AI City Challenge Multi-Camera Tracking Dataset (M4–M6)

- **Role:** Multi-camera vehicle search experiments, spatio-temporal camera transition graph estimation, candidate pruning evaluation, and test benchmarks.
- **Source:** NVIDIA AI City Challenge / University of Washington (CityFlow Benchmark)
- **Official URL:** [https://www.aicitychallenge.org/](https://www.aicitychallenge.org/)
- **License / Terms:** AI City Challenge Academic Research License Agreement (Non-commercial research use only; redistribution prohibited).
- **Access Procedure:**
  1. Register on [https://www.aicitychallenge.org/](https://www.aicitychallenge.org/) with an institutional / university email address.
  2. Accept the AI City Challenge Data License Agreement.
  3. Download Track 1 / Track 3 multi-camera tracking dataset archives (`train.zip` containing synchronized multi-camera scenarios).
- **Expected Local Structure:**
  ```
  data/raw/cityflow/
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
- **Status:** **PENDING OWNER ACTION**
  > [!NOTE]
  > Gated access requires owner acceptance of official terms. Automatic downloading or third-party mirrors without authorization are forbidden. Once placed in `data/raw/cityflow/`, the dataset loaders in `src/incident_search/io/` will parse camera calibration, synchronized timestamps, and ground-truth identities.

---

## 3. VeRi-776 Vehicle Re-Identification Dataset (M3 Baseline & Fallback)

- **Role:** Re-ID baseline evaluation benchmark (M3) and vehicle camera graph fallback.
- **Source:** Beijing University of Posts and Telecommunications (BUPT)
- **Official URL:** [https://github.com/VehicleReId/VeRi](https://github.com/VehicleReId/VeRi)
- **License / Terms:** Non-commercial Academic Research Use Only.
- **Access Procedure:**
  1. Send an email request to the authors (contact: `xinchenliu@bupt.edu.cn`).
  2. State full name, affiliation, and non-commercial research purpose.
  3. Receive official download credentials/link for `VeRi.zip`.
- **Expected Local Structure:**
  ```
  data/raw/veri776/
  ├── image_train/
  ├── image_test/
  ├── image_query/
  ├── train_label.xml
  ├── test_label.xml
  └── jk_ground_truth.txt
  ```
- **Status:** **PENDING OWNER ACTION**

---

## 4. Pretrained Vehicle Re-ID Weights (M2–M6)

- **Model:** FastReID `SBS(R50-ibn)` trained on VeRi-776
- **Architecture:** ResNet-50 with Instance-Batch Normalization (IBN) + Strong Baseline (SBS)
- **Official Source URL:** [https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth)
- **License:** Apache License 2.0 (FastReID open-source release)
- **Local Cache Path:** `outputs/weights/veri_sbs_R50-ibn.pth` (gitignored, downloaded automatically with checksum verification)
- **Performance:** 97.0% Rank-1, 81.9% mAP on VeRi-776 benchmark
- **Status:** **ACTIVE / VERIFIED** (HTTP 200, 198,261,759 bytes verified)
