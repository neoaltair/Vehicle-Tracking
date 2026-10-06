"""Unit tests for Re-ID embeddings, aggregation, and caching."""

import tempfile

import numpy as np
import pytest

from incident_search.reid.extractor import (
    FastReIDExtractor,
    aggregate_embeddings,
    l2_normalize,
    load_embeddings,
    save_embeddings,
)


def test_l2_normalize():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = l2_normalize(vec)
    assert pytest.approx(np.linalg.norm(normed), rel=1e-5) == 1.0
    assert pytest.approx(normed[0], rel=1e-5) == 0.6
    assert pytest.approx(normed[1], rel=1e-5) == 0.8


def test_aggregate_embeddings_single():
    # 3 frames, dimension 4
    feats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    agg_0 = aggregate_embeddings(feats, method="single", best_idx=0)
    assert np.allclose(agg_0, [1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(np.linalg.norm(agg_0), rel=1e-5) == 1.0

    agg_2 = aggregate_embeddings(feats, method="single", best_idx=2)
    assert np.allclose(agg_2, [0.0, 0.0, 1.0, 0.0])
    assert pytest.approx(np.linalg.norm(agg_2), rel=1e-5) == 1.0


def test_aggregate_embeddings_mean():
    feats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Mean is [0.5, 0.5, 0, 0], normalized is [1/sqrt(2), 1/sqrt(2), 0, 0]
    agg_mean = aggregate_embeddings(feats, method="mean")
    assert pytest.approx(np.linalg.norm(agg_mean), rel=1e-5) == 1.0
    val = 1.0 / np.sqrt(2.0)
    assert pytest.approx(agg_mean[0], rel=1e-5) == val
    assert pytest.approx(agg_mean[1], rel=1e-5) == val


def test_aggregate_invalid_method():
    feats = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="strictly 'single' or 'mean'"):
        aggregate_embeddings(feats, method="median")


def test_save_and_load_embeddings():
    frame_feats = np.random.randn(5, 2048).astype(np.float32)
    trk_feat = l2_normalize(np.mean(frame_feats, axis=0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_embeddings(tmpdir, "trk_test_01", frame_feats, trk_feat)
        assert path.exists()

        loaded = load_embeddings(tmpdir, "trk_test_01")
        assert loaded is not None
        loaded_frames, loaded_trk = loaded

        assert loaded_frames.shape == (5, 2048)
        assert loaded_trk.shape == (2048,)
        assert np.allclose(loaded_trk, trk_feat, atol=1e-6)


def test_fastreid_extractor_norm():
    extractor = FastReIDExtractor(device="cpu")
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_crop[20:80, 20:80] = 200  # white square

    emb = extractor.embed_crops([dummy_crop])
    assert emb.shape == (1, 2048)
    assert pytest.approx(np.linalg.norm(emb[0]), rel=1e-5) == 1.0
