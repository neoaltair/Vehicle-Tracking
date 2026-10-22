"""Download and prepare the full CityFlow AIC21 MTMC dataset for SENTINEL.

Downloads the official aic21mtmct_vehicle.zip (3.84 GB) from PaddleDetection's
Baidu BOS bucket. This contains 6 multi-camera scenarios (S01-S06) with 46
cameras total, where the same vehicles drive through multiple camera views.

Dataset structure after extraction:
    data/raw/aic21mtmct_vehicle/
    ├── test/
    │   └── S06/
    │       ├── c041/ ... c046/
    │       │   └── vdo.avi
    ├── train/
    │   ├── S01/
    │   │   ├── c001/ ... c005/
    │   │   │   ├── vdo.avi
    │   │   │   ├── roi.jpg
    │   │   │   └── gt/gt.txt
    │   │   └── map.png
    │   ├── S03/
    │   │   ├── c010/ ... c015/
    │   └── S04/
    │       ├── c016/ ... c040/
    └── validation/
        ├── S02/
        │   ├── c006/ ... c009/
        └── S05/
            └── (cameras)

Usage:
    python scripts/download_cityflow.py
    python scripts/download_cityflow.py --scenario S01   # Download + show scenario info
    python scripts/download_cityflow.py --list            # List all scenarios after download
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_URL = "https://bj.bcebos.com/v1/paddledet/data/mot/aic21mtmct_vehicle.zip"
DATASET_DIR = Path("data/raw/aic21mtmct_vehicle")
ZIP_PATH = Path("data/raw/aic21mtmct_vehicle.zip")

# Also download the small demo pair (always available as fallback)
DEMO_URL = "https://paddledet.bj.bcebos.com/data/mot/demo/mtmct-demo.tar"
DEMO_DIR = Path("data/raw/mtmct-demo")

# Scenario metadata
SCENARIOS = {
    "S01": {
        "split": "train",
        "cameras": ["c001", "c002", "c003", "c004", "c005"],
        "location": "Bellevue NE 148th Intersection Corridor",
        "description": "5 cameras covering a multi-road intersection corridor. Vehicles transition between adjacent cameras.",
    },
    "S02": {
        "split": "validation",
        "cameras": ["c006", "c007", "c008", "c009"],
        "location": "Urban arterial road network",
        "description": "4 cameras on an urban road. Validation split for metrics.",
    },
    "S03": {
        "split": "train",
        "cameras": ["c010", "c011", "c012", "c013", "c014", "c015"],
        "location": "Multi-lane highway interchange area",
        "description": "6 cameras covering highway ramps and intersections.",
    },
    "S04": {
        "split": "train",
        "cameras": [f"c{i:03d}" for i in range(16, 41)],
        "location": "Large-scale urban grid (25 cameras)",
        "description": "25 cameras across a wide urban area. Largest scenario.",
    },
    "S05": {
        "split": "validation",
        "cameras": [],  # Variable
        "location": "Urban corridor",
        "description": "Validation scenario.",
    },
    "S06": {
        "split": "test",
        "cameras": ["c041", "c042", "c043", "c044", "c045", "c046"],
        "location": "Test intersection network",
        "description": "6 cameras for official test evaluation. No ground-truth published.",
    },
}


def _download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "SENTINEL-Console/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    total = int(resp.headers.get("Content-Length", 0))
    total_mb = total / (1024 * 1024)

    logger.info("%s: %.1f MB from %s", desc, total_mb, url)

    downloaded = 0
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start_time
            speed = downloaded / (1024 * 1024 * max(elapsed, 0.01))
            pct = (downloaded / total * 100) if total else 0
            eta = ((total - downloaded) / (downloaded / max(elapsed, 0.01))) if downloaded > 0 and total > 0 else 0
            sys.stdout.write(
                f"\r  {desc}: {downloaded / (1024*1024):.1f}/{total_mb:.1f} MB "
                f"({pct:.1f}%) | {speed:.1f} MB/s | ETA: {eta:.0f}s"
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / (1024 * 1024))


def download_full_cityflow() -> Path:
    """Download and extract the full CityFlow AIC21 MTMC dataset."""
    # Check if already extracted
    if DATASET_DIR.exists():
        # Verify at least one scenario has video files
        for scenario in ["S01", "S03", "S06"]:
            for split in ["train", "validation", "test"]:
                check_dir = DATASET_DIR / split / scenario
                if check_dir.exists():
                    videos = list(check_dir.rglob("vdo.avi"))
                    if videos:
                        logger.info("CityFlow dataset already extracted at %s (%d videos found)", DATASET_DIR, len(videos))
                        return DATASET_DIR

    # Download zip
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 1_000_000_000:
        _download_with_progress(DATASET_URL, ZIP_PATH, "CityFlow AIC21 Dataset")
    else:
        logger.info("ZIP already downloaded: %s (%.2f GB)", ZIP_PATH, ZIP_PATH.stat().st_size / (1024**3))

    # Extract
    logger.info("Extracting dataset (this may take several minutes)...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(ZIP_PATH), "r") as zf:
        # Check what's inside
        top_dirs = {name.split("/")[0] for name in zf.namelist() if "/" in name}
        logger.info("Archive contains top-level directories: %s", top_dirs)
        zf.extractall(str(DATASET_DIR.parent))

    # The zip might extract as AIC21_Track3_MTMC_Tracking/ — handle both cases
    alt_dir = DATASET_DIR.parent / "AIC21_Track3_MTMC_Tracking"
    if alt_dir.exists() and not (DATASET_DIR / "train").exists():
        logger.info("Renaming %s -> %s", alt_dir, DATASET_DIR)
        if DATASET_DIR.exists():
            import shutil
            shutil.rmtree(DATASET_DIR)
        alt_dir.rename(DATASET_DIR)

    logger.info("Extraction complete!")
    return DATASET_DIR


def list_scenarios(base_dir: Path) -> dict[str, dict]:
    """Scan the extracted dataset and list available scenarios with camera counts."""
    found = {}
    for split in ["train", "validation", "test"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        for scenario_dir in sorted(split_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario_name = scenario_dir.name
            cameras = []
            for cam_dir in sorted(scenario_dir.iterdir()):
                if cam_dir.is_dir():
                    video = cam_dir / "vdo.avi"
                    if video.exists():
                        cameras.append({
                            "camera_id": cam_dir.name,
                            "video_path": str(video),
                            "video_size_mb": video.stat().st_size / (1024 * 1024),
                            "has_gt": (cam_dir / "gt" / "gt.txt").exists(),
                        })
            if cameras:
                found[scenario_name] = {
                    "split": split,
                    "path": str(scenario_dir),
                    "num_cameras": len(cameras),
                    "cameras": cameras,
                    "total_video_mb": sum(c["video_size_mb"] for c in cameras),
                    "has_ground_truth": any(c["has_gt"] for c in cameras),
                }
    return found


def print_scenario_summary(scenarios: dict[str, dict]) -> None:
    """Print a formatted summary of available scenarios."""
    print("\n" + "=" * 80)
    print("  CITYFLOW AIC21 MTMC DATASET — AVAILABLE SCENARIOS")
    print("=" * 80)

    for name, info in sorted(scenarios.items()):
        meta = SCENARIOS.get(name, {})
        print(f"\n  📁 {name} ({info['split'].upper()})")
        print(f"     Location:     {meta.get('location', 'Unknown')}")
        print(f"     Description:  {meta.get('description', 'N/A')}")
        print(f"     Cameras:      {info['num_cameras']}")
        print(f"     Total Video:  {info['total_video_mb']:.1f} MB")
        print(f"     Ground Truth: {'✅ Yes' if info['has_ground_truth'] else '❌ No'}")
        cam_ids = [c["camera_id"] for c in info["cameras"]]
        print(f"     Camera IDs:   {', '.join(cam_ids)}")

    total_cams = sum(s["num_cameras"] for s in scenarios.values())
    total_mb = sum(s["total_video_mb"] for s in scenarios.values())
    print(f"\n  {'─' * 60}")
    print(f"  TOTAL: {len(scenarios)} scenarios, {total_cams} cameras, {total_mb:.0f} MB video")
    print("=" * 80 + "\n")


def get_best_demo_scenarios(base_dir: Path) -> list[dict]:
    """Return the best scenarios for a convincing cross-camera demo, ranked."""
    scenarios = list_scenarios(base_dir)
    recommended = []

    # S01 is the best — 5 cameras, ground truth, intersection corridor
    if "S01" in scenarios:
        s = scenarios["S01"]
        recommended.append({
            "scenario": "S01",
            "reason": "BEST FOR DEMO: 5 cameras at Bellevue NE 148th corridor, ground-truth labels, adjacent road cameras",
            "cameras": s["cameras"],
            "suggested_main": "c001",
            "suggested_bg": ["c002", "c003", "c004", "c005"],
        })

    # S03 — 6 cameras, highway interchange
    if "S03" in scenarios:
        s = scenarios["S03"]
        recommended.append({
            "scenario": "S03",
            "reason": "6 cameras at highway interchange, good variety of vehicle transitions",
            "cameras": s["cameras"],
            "suggested_main": "c010",
            "suggested_bg": ["c011", "c012", "c013", "c014", "c015"],
        })

    # S02 — 4 cameras, validation
    if "S02" in scenarios:
        s = scenarios["S02"]
        recommended.append({
            "scenario": "S02",
            "reason": "4 cameras on urban arterial road, validation split",
            "cameras": s["cameras"],
            "suggested_main": "c006",
            "suggested_bg": ["c007", "c008", "c009"],
        })

    return recommended


def main():
    parser = argparse.ArgumentParser(description="Download CityFlow AIC21 MTMC dataset for SENTINEL")
    parser.add_argument("--list", action="store_true", help="List available scenarios after download")
    parser.add_argument("--scenario", type=str, help="Show detailed info for a specific scenario (e.g. S01)")
    parser.add_argument("--recommend", action="store_true", help="Show recommended demo configurations")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just scan existing data")
    args = parser.parse_args()

    if not args.skip_download:
        download_full_cityflow()

    if args.list or args.scenario or args.recommend:
        if not DATASET_DIR.exists():
            logger.error("Dataset directory not found: %s", DATASET_DIR)
            sys.exit(1)

        scenarios = list_scenarios(DATASET_DIR)
        if not scenarios:
            logger.error("No scenarios found in %s", DATASET_DIR)
            sys.exit(1)

        if args.list:
            print_scenario_summary(scenarios)

        if args.scenario:
            s = scenarios.get(args.scenario.upper())
            if s:
                print(f"\nScenario {args.scenario.upper()}:")
                for cam in s["cameras"]:
                    print(f"  Camera {cam['camera_id']}: {cam['video_size_mb']:.1f} MB | GT: {cam['has_gt']}")
            else:
                print(f"Scenario {args.scenario} not found. Available: {list(scenarios.keys())}")

        if args.recommend:
            recs = get_best_demo_scenarios(DATASET_DIR)
            print("\n🏆 RECOMMENDED DEMO CONFIGURATIONS:")
            for i, r in enumerate(recs, 1):
                print(f"\n  #{i} Scenario {r['scenario']}: {r['reason']}")
                print(f"     Main Camera: {r['suggested_main']}")
                print(f"     Background:  {', '.join(r['suggested_bg'])}")


if __name__ == "__main__":
    main()
