"""Tests for BraTS advanced post-processing pipeline."""

import numpy as np
import os
import importlib.util


_pp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "utils", "postprocess.py")
_spec = importlib.util.spec_from_file_location("postprocess", _pp_path)
_pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pp)

postprocess_brats_advanced = _pp.postprocess_brats_advanced
postprocess_et = _pp.postprocess_et
fill_holes_et = _pp.fill_holes_et


def _make_volume(shape=(32, 32, 32), fill=0):
    return np.full(shape, fill, dtype=np.int64)


def _place_sphere(vol, center, radius, label):
    z, y, x = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    vol[dist <= radius] = label
    return vol


class TestPostprocessBratsAdvanced:
    """Test the championship-level post-processing pipeline."""

    def test_empty_prediction(self):
        """All background → no change."""
        pred = _make_volume()
        result = postprocess_brats_advanced(pred)
        assert np.array_equal(result, pred)

    def test_small_wt_removed(self):
        """Small WT island (< wt_min_size) should be removed."""
        pred = _make_volume()
        pred[5:7, 5:10, 5:10] = 2  # edema, 50 voxels
        result = postprocess_brats_advanced(pred, wt_min_size=500)
        assert result.sum() == 0, "Small WT island should be removed"

    def test_large_wt_kept(self):
        """Large WT island (>= wt_min_size) should be kept."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 8, 2)  # large edema sphere
        result = postprocess_brats_advanced(pred, wt_min_size=500)
        assert (result > 0).sum() > 0, "Large WT should be kept"

    def test_small_et_replaced_with_ncr(self):
        """Small ET component should be reclassified as NCR (label 1)."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 8, 2)  # edema
        _place_sphere(pred, (16, 16, 16), 5, 1)  # NCR core
        pred[16:18, 16:18, 16:18] = 3  # 8 voxels of ET
        result = postprocess_brats_advanced(pred, et_min_size=200)
        assert (result == 3).sum() == 0, "Small ET should be removed"

    def test_large_et_kept(self):
        """Large ET component (>= et_min_size) should be kept."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)  # edema
        _place_sphere(pred, (32, 32, 32), 10, 1)  # NCR
        _place_sphere(pred, (32, 32, 32), 7, 3)   # ET (~1437 voxels)
        result = postprocess_brats_advanced(pred, et_min_size=200)
        assert (result == 3).sum() > 0, "Large ET should be kept"

    def test_small_tc_becomes_edema(self):
        """Small TC component should be reclassified as edema."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 10, 2)  # large edema
        pred[5, 5, 5] = 1  # isolated small NCR spot
        result = postprocess_brats_advanced(pred, tc_min_size=200)
        assert result[5, 5, 5] == 2 or result[5, 5, 5] == 0, \
            "Small isolated TC should be reclassified"

    def test_wt_hole_filling(self):
        """Internal background holes in WT should be filled with edema."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 10, 2)  # edema shell
        pred[15:18, 15:18, 15:18] = 0  # punch hole
        result = postprocess_brats_advanced(pred, fill_wt_holes=True)
        assert (result[15:18, 15:18, 15:18] == 2).all(), "WT hole should be filled"

    def test_et_hole_filling(self):
        """Internal holes in ET should be filled."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)
        _place_sphere(pred, (32, 32, 32), 10, 1)
        _place_sphere(pred, (32, 32, 32), 7, 3)
        pred[32, 32, 32] = 1  # NCR hole in ET
        result = postprocess_brats_advanced(pred, et_min_size=50, fill_et_holes=True)
        assert result[32, 32, 32] == 3, "Hole inside ET should be filled"

    def test_hierarchy_isolated_et_in_edema(self):
        """ET floating in pure edema (no NCR neighbor) should be reclassified."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 10, 2)  # edema only
        pred[16:19, 16:19, 16:19] = 3  # 27 voxels ET, no NCR nearby
        result = postprocess_brats_advanced(pred, et_min_size=10)
        assert (result == 3).sum() == 0, "Isolated ET in edema should be removed"

    def test_et_with_ncr_neighbor_kept(self):
        """ET adjacent to NCR should be kept."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)
        _place_sphere(pred, (32, 32, 32), 10, 1)
        _place_sphere(pred, (32, 32, 32), 7, 3)
        et_before = (pred == 3).sum()
        result = postprocess_brats_advanced(pred, et_min_size=50)
        et_after = (result == 3).sum()
        assert et_after > et_before * 0.8, "ET with NCR neighbor should be kept"

    def test_confidence_filtering_et(self):
        """Low-confidence ET should be removed when threshold set."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)
        _place_sphere(pred, (32, 32, 32), 10, 1)
        _place_sphere(pred, (32, 32, 32), 7, 3)

        probs = np.zeros((4, 64, 64, 64), dtype=np.float32)
        probs[0] = 0.1
        probs[2] = 0.5
        probs[1] = 0.3
        probs[3] = 0.1  # ET very low confidence

        result = postprocess_brats_advanced(
            pred, softmax_probs=probs, et_min_size=50, et_conf_thresh=0.3
        )
        assert (result == 3).sum() == 0, "Low-confidence ET should be filtered"

    def test_confidence_keeps_high_conf(self):
        """High-confidence ET should be kept."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)
        _place_sphere(pred, (32, 32, 32), 10, 1)
        _place_sphere(pred, (32, 32, 32), 7, 3)

        probs = np.zeros((4, 64, 64, 64), dtype=np.float32)
        probs[3] = 0.8

        result = postprocess_brats_advanced(
            pred, softmax_probs=probs, et_min_size=50, et_conf_thresh=0.3
        )
        assert (result == 3).sum() > 0, "High-confidence ET should be kept"

    def test_no_fill_holes_option(self):
        """Disabling hole filling should leave holes intact."""
        pred = _make_volume()
        _place_sphere(pred, (16, 16, 16), 10, 2)
        pred[16, 16, 16] = 0
        result = postprocess_brats_advanced(pred, fill_wt_holes=False, fill_et_holes=False)
        assert result[16, 16, 16] == 0, "Hole should remain when filling disabled"

    def test_pipeline_preserves_valid_prediction(self):
        """Well-formed prediction should pass through mostly unchanged."""
        pred = _make_volume(shape=(64, 64, 64))
        _place_sphere(pred, (32, 32, 32), 15, 2)
        _place_sphere(pred, (32, 32, 32), 10, 1)
        _place_sphere(pred, (32, 32, 32), 7, 3)

        result = postprocess_brats_advanced(pred, wt_min_size=100, tc_min_size=50, et_min_size=50)
        assert (result == 1).sum() > 0
        assert (result == 2).sum() > 0
        assert (result == 3).sum() > 0


class TestLegacyPostprocess:
    """Verify legacy functions still work."""

    def test_postprocess_et_removes_small(self):
        pred = _make_volume()
        pred[5:7, 5:7, 5:7] = 3  # 8 voxels
        result = postprocess_et(pred, min_size=50)
        assert (result == 3).sum() == 0
        assert (result == 1).sum() == 8

    def test_fill_holes_et(self):
        pred = _make_volume(shape=(32, 32, 32))
        _place_sphere(pred, (16, 16, 16), 5, 3)
        pred[16, 16, 16] = 0
        result = fill_holes_et(pred)
        assert result[16, 16, 16] == 3
