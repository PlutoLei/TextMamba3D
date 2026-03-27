from autoresearch.hypothesis_engine import format_results_for_analysis

def test_format_results():
    results = {
        'baseline': {'mean_dice': 0.8479, 'dice_ET': 0.7910, 'dice_TC': 0.856, 'dice_WT': 0.8967},
        'experiments': [
            {'id': 'L0-1', 'metrics': {'dice_mean': 0.8500, 'dice_ET': 0.8000}, 'improved': True},
        ]
    }
    prompt = format_results_for_analysis(results)
    assert 'baseline' in prompt.lower()
    assert 'L0-1' in prompt
    assert '0.8500' in prompt
    assert 'IMPROVED' in prompt
