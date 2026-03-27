"""AutoResearch orchestrator: manage experiment lifecycle."""
import json
from datetime import datetime


class Orchestrator:
    def __init__(self, experiments_path: str, results_path: str):
        self.experiments_path = experiments_path
        self.results_path = results_path
        with open(experiments_path) as f:
            self.experiments = json.load(f)
        with open(results_path) as f:
            self.results = json.load(f)

    def next_experiment(self) -> dict | None:
        queue = self.experiments['queue']
        if not queue:
            return None
        queue.sort(key=lambda x: x['layer'])
        return queue[0]

    def record_result(self, experiment_id: str, metrics: dict) -> None:
        result = {
            'id': experiment_id,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'improved': self.is_improvement(metrics),
        }
        self.results['experiments'].append(result)
        if self.is_improvement(metrics):
            self.results['best'] = {
                'mean_dice': metrics['dice_mean'],
                'experiment_id': experiment_id,
            }
        self.experiments['queue'] = [
            e for e in self.experiments['queue'] if e['id'] != experiment_id
        ]
        self.experiments.setdefault('completed', []).append({
            'id': experiment_id, 'result': result
        })
        self._save()

    def is_improvement(self, metrics: dict) -> bool:
        return metrics.get('dice_mean', 0) > self.results['best']['mean_dice']

    def consecutive_finetune_failures(self) -> int:
        count = 0
        for exp in reversed(self.results['experiments']):
            if exp.get('id', '').startswith('L1') and not exp.get('improved'):
                count += 1
            else:
                break
        return count

    def should_train_from_scratch(self) -> bool:
        return self.consecutive_finetune_failures() >= 3

    def status(self) -> str:
        queue_by_layer = {}
        for e in self.experiments['queue']:
            queue_by_layer.setdefault(e['layer'], []).append(e)
        lines = [
            f"Best: {self.results['best']['mean_dice']:.4f} ({self.results['best']['experiment_id']})",
            f"Queue: {len(self.experiments['queue'])} experiments",
        ]
        for layer in sorted(queue_by_layer):
            lines.append(f"  L{layer}: {len(queue_by_layer[layer])}")
        lines.append(f"Completed: {len(self.results['experiments'])} experiments")
        improvements = sum(1 for e in self.results['experiments'] if e.get('improved'))
        lines.append(f"Improvements: {improvements}")
        return '\n'.join(lines)

    def _save(self) -> None:
        with open(self.experiments_path, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
