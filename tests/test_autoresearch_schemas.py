import json

def test_experiments_json_valid():
    with open('autoresearch/experiments.json') as f:
        data = json.load(f)
    assert 'queue' in data
    assert 'completed' in data
    assert isinstance(data['queue'], list)
    for exp in data['queue']:
        assert 'id' in exp
        assert 'layer' in exp
        assert exp['layer'] in [0, 1, 2]
        assert 'type' in exp
        assert 'params' in exp

def test_results_json_valid():
    with open('autoresearch/results.json') as f:
        data = json.load(f)
    assert 'baseline' in data
    assert data['baseline']['mean_dice'] == 0.8479
    assert 'experiments' in data
    assert isinstance(data['experiments'], list)
