from autoresearch.result_collector import parse_eval_output, is_improvement

SAMPLE_OUTPUT = """
============================================================
Results: test split, 95 cases, text=True
============================================================
  dice_ET: 0.7910 +/- 0.2300
  dice_TC: 0.8560 +/- 0.1710
  dice_WT: 0.8967 +/- 0.1021
  dice_mean: 0.8479 +/- 0.1321
  hd95_ET: 3.50 +/- 17.29
  hd95_TC: 2.39 +/- 17.49
  hd95_WT: 2.91 +/- 13.49
============================================================
"""

def test_parse_eval_output():
    metrics = parse_eval_output(SAMPLE_OUTPUT)
    assert metrics['dice_ET'] == 0.7910
    assert metrics['dice_TC'] == 0.8560
    assert metrics['dice_WT'] == 0.8967
    assert metrics['dice_mean'] == 0.8479

def test_parse_empty():
    metrics = parse_eval_output("no metrics here")
    assert metrics == {}

def test_is_improvement_true():
    assert is_improvement({'dice_mean': 0.85}) is True

def test_is_improvement_false():
    assert is_improvement({'dice_mean': 0.84}) is False
