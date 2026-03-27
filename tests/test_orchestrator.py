import json, os, tempfile, shutil
from autoresearch.orchestrator import Orchestrator

def _setup():
    tmpdir = tempfile.mkdtemp()
    exp_path = os.path.join(tmpdir, 'experiments.json')
    res_path = os.path.join(tmpdir, 'results.json')
    shutil.copy('autoresearch/experiments.json', exp_path)
    shutil.copy('autoresearch/results.json', res_path)
    return tmpdir, exp_path, res_path

def test_next_experiment_returns_l0_first():
    tmpdir, ep, rp = _setup()
    orch = Orchestrator(ep, rp)
    exp = orch.next_experiment()
    assert exp is not None
    assert exp['layer'] == 0
    shutil.rmtree(tmpdir)

def test_record_result():
    tmpdir, ep, rp = _setup()
    orch = Orchestrator(ep, rp)
    exp = orch.next_experiment()
    orch.record_result(exp['id'], {'dice_mean': 0.85, 'dice_ET': 0.80})
    assert len(orch.results['experiments']) == 1
    assert orch.results['best']['mean_dice'] == 0.85
    shutil.rmtree(tmpdir)

def test_is_improvement():
    tmpdir, ep, rp = _setup()
    orch = Orchestrator(ep, rp)
    assert orch.is_improvement({'dice_mean': 0.85}) is True
    assert orch.is_improvement({'dice_mean': 0.84}) is False
    shutil.rmtree(tmpdir)

def test_status():
    tmpdir, ep, rp = _setup()
    orch = Orchestrator(ep, rp)
    s = orch.status()
    assert 'Best:' in s
    assert 'Queue:' in s
    shutil.rmtree(tmpdir)
