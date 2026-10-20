"""Replay every canonical artifact comparison using NumPy and the core CLI only.

No Torch, native framework, accelerator, pretrained checkpoint or pickle is
loaded. Reassembled safe-tensor artifacts retain their native original dtypes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    prefix = chr(92) * 2 + '?' + chr(92)
    if os.name == 'nt' and not str(root).startswith(prefix):
        root = Path(prefix + str(root))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads(Path(a.index).read_text())
    core = root / 'metadata/core-tools'
    records = []
    for case in index['cases']:
        if not case['canonical']: continue
        folder = root / Path(*case['logical_root'].strip('/').split('/')) / 'metrics'
        for stage in ['gradients', 'updates', 'observations', 'distributions']:
            script = core / ('compare_distributions.py' if stage == 'distributions' else 'compare_tensors.py')
            target = out / (case['label'] + '-' + stage + '.json')
            command = [sys.executable, '-B', str(script), '--contract', str(folder / (stage + '-contract.json')), '--side-a', str(folder / (stage + '-native.json')), '--side-b', str(folder / (stage + '-reference.json')), '--output', str(target)]
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (out / (target.stem + '.log')).write_text(completed.stdout)
            assert completed.returncode == 0, (case['label'], stage, completed.stdout)
            result = json.loads(target.read_text())
            assert result['status'] == 'no_thresholds', (case['label'], stage, result['status'])
            records.append({'case': case['label'], 'stage': stage, 'returncode': completed.returncode, 'status': result['status'], 'core_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    proof = {'status': 'all_canonical_core_comparisons_replayed', 'canonical_cases': sum(c['canonical'] for c in index['cases']), 'comparisons': len(records), 'torch_imported': 'torch' in sys.modules, 'pickle_loaded': False, 'thresholds': None, 'records': records}
    assert not proof['torch_imported'] and len(records) == 4 * proof['canonical_cases']
    (out / 'replay-completion.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__': main()
