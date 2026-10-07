# Limitations and Ethics

## Known Technical Limitations

### Dataset scope
- **CityFlow v2** and **VeRi-776** are the only ground-truth datasets planned for
  evaluation. Both cover limited geographies and daylight/controlled-lighting
  conditions. Results may not generalise to night-time, adverse weather, or cameras
  with very different field-of-view characteristics.
- Ground-truth tracklets are used for the controlled experiment. Detector/tracker
  failures in the YOLO+ByteTrack pipeline are an additional noise source that is
  intentionally separated from the main research question.

### Single-camera temporal constraint
- The upstream temporal window `[Tq - T, Tq]` relies on a calibrated global clock.
  CityFlow v2 provides synchronised video; real deployments may have clock drift.
  Clock error is not modelled in this study.

### Re-ID model
- FastReID SBS R50-ibn was pretrained on VeRi-776 training identities. Evaluating
  the same model on the VeRi-776 test split creates mild distribution overlap.
  For the cleanest baseline, use the cross-dataset setting (train on VeRi-776,
  evaluate on CityFlow v2) where possible.
- The model does not use licence plates, colour histograms, or any modality beyond
  the raw RGB crop. Low-resolution or heavily occluded queries degrade embedding
  quality; the degradation study measures this directly.

### Camera graph statistics
- Travel-time priors are estimated from training-split identities only. Sparse
  edges (few observed transitions) produce unreliable log-normal fits. The
  `min_edge_samples` threshold in `configs/default.yaml` guards against this; edges
  below the threshold should be treated as uninformative and either dropped or
  assigned a flat uniform prior.

### Scale
- All retrieval uses exact cosine similarity on CPU. Gallery size is bounded by
  the dataset. FAISS indexing is deferred; latency benchmarks at M4 will determine
  whether it is necessary.

---

## Ethical and Misuse Risks

### Surveillance and privacy
- This system is designed to assist post-incident forensic investigation of traffic
  incidents using datasets whose terms explicitly permit research use. It is **not**
  intended for continuous mass surveillance, real-time tracking of individuals, or
  any purpose inconsistent with the originating dataset licences.
- The system processes vehicle appearance only; it is not designed to identify
  drivers or passengers.

### Human-in-the-loop requirement
- The system produces a **ranked candidate list, never a declared identity**.
  Every output is labelled with its similarity score and must be reviewed and
  confirmed by a qualified human operator before any action is taken.
- Automated use of the ranked list to trigger enforcement decisions is out of scope
  and would require separate ethical and legal review.

### Bias and fairness
- Re-ID models trained on specific vehicle datasets may perform unequally across
  vehicle colours, models, or makes that are under-represented in the training set.
- The degradation study isolates image-quality effects; systematic demographic or
  vehicle-type biases in the re-ID model are an acknowledged limitation and a topic
  for future work.

### Data handling
- No personally identifiable information (PII) is stored by the pipeline.
- Raw video, model weights, and dataset files are gitignored and must not be
  committed or shared beyond the terms of the dataset licence agreements.
- See `docs/DATA.md` for each dataset's licence and access conditions.

---

## Scope Statement

This is a **research prototype** for a student project. It is not validated for
production use, law-enforcement deployment, or any safety-critical application.
