# tests/test_et_quantitative_text.py
"""Tests for quantitative ET text generation."""
import numpy as np
import os
import importlib.util

# Direct import to avoid torch dependency (same pattern as test_postprocess.py)
_mod_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "et_quantitative_text.py")
_spec = importlib.util.spec_from_file_location("et_quantitative_text", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_et_stats = _mod.compute_et_stats
generate_quantitative_et_text = _mod.generate_quantitative_et_text


def _make_volume(shape=(64, 64, 64), fill=0):
    return np.full(shape, fill, dtype=np.int64)

def _place_sphere(vol, center, radius, label):
    z, y, x = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    vol[dist <= radius] = label
    return vol


class TestComputeEtStats:
    def test_no_tumor(self):
        mask = _make_volume()
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] == 0
        assert stats["et_ratio"] == 0.0

    def test_with_et(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        _place_sphere(mask, (32, 32, 32), 10, 1)
        _place_sphere(mask, (32, 32, 32), 7, 3)
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] > 0
        assert 0 < stats["et_ratio"] < 1.0
        assert stats["n_et_clusters"] >= 1
        assert stats["side"] in ("left", "right")
        assert "position" in stats

    def test_small_et(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        _place_sphere(mask, (32, 32, 32), 10, 1)
        mask[32, 32, 32] = 3
        stats = compute_et_stats(mask)
        assert stats["et_voxels"] == 1
        assert stats["size_category"] == "minimal"

    def test_multiple_clusters(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        mask[10:13, 10:13, 10:13] = 3
        mask[50:53, 50:53, 50:53] = 3
        stats = compute_et_stats(mask)
        assert stats["n_et_clusters"] == 2


class TestGenerateQuantitativeEtText:
    def test_no_et_text(self):
        mask = _make_volume()
        text = generate_quantitative_et_text(mask)
        assert "absent" in text.lower()

    def test_with_et_text(self):
        mask = _make_volume()
        _place_sphere(mask, (32, 32, 32), 15, 2)
        _place_sphere(mask, (32, 32, 32), 10, 1)
        _place_sphere(mask, (32, 32, 32), 7, 3)
        text = generate_quantitative_et_text(mask)
        assert "%" in text
        assert len(text) > 20

    def test_text_varies_with_et_size(self):
        mask1 = _make_volume()
        _place_sphere(mask1, (32, 32, 32), 15, 2)
        mask1[32, 32, 32] = 3

        mask2 = _make_volume()
        _place_sphere(mask2, (32, 32, 32), 15, 2)
        _place_sphere(mask2, (32, 32, 32), 10, 1)
        _place_sphere(mask2, (32, 32, 32), 7, 3)

        text1 = generate_quantitative_et_text(mask1)
        text2 = generate_quantitative_et_text(mask2)
        assert text1 != text2
