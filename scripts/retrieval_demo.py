"""retrieval_demo.py – Embed all tracklets from an M1 run and visualise top-5 matches.

Usage (from repo root):
    python scripts/retrieval_demo.py [--tracklets PATH] [--weights PATH] [--out PATH]

Outputs
-------
docs/assets/retrieval_demo_top5.png   query | top-5 match image grid
outputs/cache/embeddings/             per-tracklet .npz embedding cache

Algorithm
---------
1. Load all tracklets from the JSON produced by run_detect_track.py.
2. For each tracklet, embed all crops with FastReIDExtractor.embed_crops() and
   mean-pool to one 2048-d unit-norm vector.  Cache each tracklet embedding with
   extractor.save_embeddings() so subsequent runs skip inference.
3. Query = tracklet with the most frames (strongest temporal signal).
4. Gallery = all other tracklets.
5. Rank by cosine similarity (L2-normed vectors → dot product = cosine).
6. Build a 1×(1+K) grid: query cell | top-K match cells with labels + scores.
7. Save to docs/assets/retrieval_demo_top5.png.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Resolve package root so the script works from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from incident_search.io.schema import Tracklet, load_tracklets  # noqa: E402
from incident_search.reid.extractor import (  # noqa: E402
    FastReIDExtractor,
    aggregate_embeddings,
    load_embeddings,
    save_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
THUMB_W = 160
THUMB_H = 200
LABEL_H = 34
PAD = 6
BG_COLOR = (30, 30, 30)
LABEL_COLOR = (255, 255, 255)
QUERY_BORDER = (255, 215, 0)    # gold
MATCH_BORDER = (100, 200, 100)  # green


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _best_crop_path(tracklet: Tracklet) -> str | None:
    """Return the middle crop path (most representative frame)."""
    paths = [p for p in tracklet.crop_paths if Path(p).exists()]
    if not paths:
        return None
    return paths[len(paths) // 2]


def _make_thumb(img_path: str | None, label: str, border_color: tuple) -> Image.Image:
    """Return a fixed-size PIL thumbnail cell with a label bar at the bottom."""
    cell_w = THUMB_W + 2 * PAD
    cell_h = THUMB_H + LABEL_H + 3 * PAD
    cell = Image.new("RGB", (cell_w, cell_h), BG_COLOR)
    draw = ImageDraw.Draw(cell)

    # Paste crop thumbnail
    if img_path and Path(img_path).exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
            x_off = PAD + (THUMB_W - img.width) // 2
            y_off = PAD + (THUMB_H - img.height) // 2
            cell.paste(img, (x_off, y_off))
        except Exception:
            pass  # leave blank on failure

    # Coloured border around thumb area
    draw.rectangle(
        [PAD - 1, PAD - 1, PAD + THUMB_W, PAD + THUMB_H],
        outline=border_color,
        width=2,
    )

    # Label bar
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    label_y = THUMB_H + 2 * PAD
    draw.rectangle([0, label_y, cell_w, cell_h], fill=(50, 50, 50))
    for i, line in enumerate(label.split("\n")):
        draw.text((PAD, label_y + 4 + i * 14), line, fill=LABEL_COLOR, font=font)

    return cell


def _build_grid(query_cell: Image.Image, match_cells: list[Image.Image]) -> Image.Image:
    """Arrange query on left, matches in a row to the right."""
    n_cols = 1 + len(match_cells)
    cell_w, cell_h = query_cell.size
    grid = Image.new("RGB", (n_cols * cell_w, cell_h), BG_COLOR)
    grid.paste(query_cell, (0, 0))
    for i, mc in enumerate(match_cells):
        grid.paste(mc, ((i + 1) * cell_w, 0))
    return grid


# ---------------------------------------------------------------------------
# Embedding pipeline (uses save_embeddings / load_embeddings from extractor.py)
# ---------------------------------------------------------------------------

def _get_tracklet_embedding(
    tracklet: Tracklet,
    extractor: FastReIDExtractor,
    cache_dir: Path,
) -> np.ndarray | None:
    """Return cached or freshly computed mean-pooled unit-norm embedding for a tracklet.

    Returns None if the tracklet has no readable crop images.

    Cache layout (per-tracklet .npz, matches save_embeddings() in extractor.py):
        outputs/cache/embeddings/sbs_R50-ibn/<tracklet_id>.npz
            frame_embeddings  [N, 2048]
            tracklet_embedding [2048]
    """
    # Try cache first
    cached = load_embeddings(str(cache_dir), tracklet.tracklet_id)
    if cached is not None:
        _frame_embs, trk_emb = cached
        return trk_emb  # already L2-normed

    # Compute from crops
    valid_paths = [p for p in tracklet.crop_paths if Path(p).exists()]
    if not valid_paths:
        log.warning("Tracklet %s: no readable crops – skipping.", tracklet.tracklet_id)
        return None

    frame_embs = extractor.embed_crops(valid_paths)          # [N, 2048]
    trk_emb = aggregate_embeddings(frame_embs, method="mean")  # [2048]

    # Persist cache
    save_embeddings(
        cache_dir=str(cache_dir),
        tracklet_id=tracklet.tracklet_id,
        frame_embeddings=frame_embs,
        tracklet_embedding=trk_emb,
    )
    log.info("  %s: %d crops → cached", tracklet.tracklet_id, len(valid_paths))
    return trk_emb


def embed_all_tracklets(
    tracklets: list[Tracklet],
    extractor: FastReIDExtractor,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """Return {tracklet_id: unit_norm_embedding} for all tracklets that have valid crops."""
    log.info("Embedding %d tracklets (cache: %s) …", len(tracklets), cache_dir)
    emb_map: dict[str, np.ndarray] = {}
    for tr in tracklets:
        emb = _get_tracklet_embedding(tr, extractor, cache_dir)
        if emb is not None:
            emb_map[tr.tracklet_id] = emb
    log.info("Embedded %d / %d tracklets.", len(emb_map), len(tracklets))
    return emb_map


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_retrieval_demo(
    tracklets_path: str,
    weights_path: str,
    out_path: str,
    top_k: int = 5,
) -> None:
    out_path_obj = Path(out_path)
    cache_dir = _REPO_ROOT / "outputs" / "cache" / "embeddings"

    # 1. Load tracklets
    log.info("Loading tracklets from %s", tracklets_path)
    tracklets: list[Tracklet] = load_tracklets(tracklets_path)
    log.info("Loaded %d tracklets.", len(tracklets))

    if len(tracklets) < 2:
        raise ValueError(
            f"Need at least 2 tracklets for retrieval demo; got {len(tracklets)}."
        )

    # 2. Build model
    log.info("Loading FastReID weights: %s", weights_path)
    extractor = FastReIDExtractor(weights_path)

    # 3. Embed all tracklets (with per-tracklet cache)
    emb_map = embed_all_tracklets(tracklets, extractor, cache_dir)

    ids = list(emb_map.keys())
    if len(ids) < 2:
        raise ValueError("Need embeddings for at least 2 tracklets; too many had no readable crops.")

    # 4. Query = tracklet with most frames
    frame_counts = {tr.tracklet_id: len(tr.frames) for tr in tracklets}
    query_id = max(ids, key=lambda tid: frame_counts.get(tid, 0))
    gallery_ids = [tid for tid in ids if tid != query_id]
    log.info(
        "Query: %s (%d frames)  Gallery size: %d",
        query_id, frame_counts.get(query_id, 0), len(gallery_ids),
    )

    # 5. Cosine similarity (L2-normed → dot product = cosine)
    q_emb = emb_map[query_id]                                    # [2048]
    g_mat = np.stack([emb_map[gid] for gid in gallery_ids])     # [G, 2048]
    sims: np.ndarray = q_emb @ g_mat.T                          # [G]
    ranked_idx = np.argsort(-sims)
    top_k_idx = ranked_idx[: min(top_k, len(ranked_idx))]

    log.info("Top-%d ranking:", len(top_k_idx))
    for rank, idx in enumerate(top_k_idx, 1):
        log.info("  #%d  %-22s  sim=%.4f", rank, gallery_ids[idx], float(sims[idx]))

    # 6. Build grid
    tr_by_id = {tr.tracklet_id: tr for tr in tracklets}

    query_label = f"QUERY\n{query_id}"
    query_cell = _make_thumb(_best_crop_path(tr_by_id[query_id]), query_label, QUERY_BORDER)

    match_cells: list[Image.Image] = []
    for rank, idx in enumerate(top_k_idx, 1):
        gid = gallery_ids[idx]
        label = f"#{rank}  sim={sims[idx]:.3f}\n{gid}"
        mc = _make_thumb(_best_crop_path(tr_by_id[gid]), label, MATCH_BORDER)
        match_cells.append(mc)

    grid = _build_grid(query_cell, match_cells)

    # 7. Save
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(out_path_obj))
    size_mb = out_path_obj.stat().st_size / 1024 / 1024
    log.info("Saved %s  (%dx%d, %.2f MB)", out_path_obj, grid.width, grid.height, size_mb)
    if size_mb > 5:
        log.warning("Output exceeds 5 MB — reduce resolution before committing.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReID retrieval demo – top-K match grid.")
    p.add_argument(
        "--tracklets",
        default=str(_REPO_ROOT / "outputs" / "m1_run" / "c001_tracklets.json"),
        help="Path to tracklets JSON (default: outputs/m1_run/c001_tracklets.json).",
    )
    p.add_argument(
        "--weights",
        default=str(_REPO_ROOT / "outputs" / "weights" / "veri_sbs_R50-ibn.pth"),
        help="Path to FastReID weights (.pth).",
    )
    p.add_argument(
        "--out",
        default=str(_REPO_ROOT / "docs" / "assets" / "retrieval_demo_top5.png"),
        help="Output PNG path.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Number of matches to show (default 5).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_retrieval_demo(
        tracklets_path=args.tracklets,
        weights_path=args.weights,
        out_path=args.out,
        top_k=args.top_k,
    )
