#!/usr/bin/env python3
"""Replay a hashed artifact bundle through the existing skill tools.

Inputs: ZIP containing EXPORT.json (files:[{path,exported_sha256,...}]) and
REPLAY.json (cases:[{name,operation,inputs,expected}]). Supported operations:
compare_tensors, compare_distributions and compare_target_scores with
contract/side_a/side_b paths; validate with contract/manifest.
Paths are relative to the extracted archive. Expected is a nested result subset.
Outputs: a fresh compare-replay-UTC directory with inputs, per-case JSON reports
and replay.json; exit 0 if all replay assertions match, otherwise 1.
Requires NumPy through the skill tools. This does not collect model outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe_path(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('invalid bundle path: '+name)
    result = root.joinpath(*path.parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('bundle path leaves input directory')
    return result


def check_subset(actual, expected, label='result'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(label+': expected object')
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(label+'.'+key+': missing')
            check_subset(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(label+': list length/type differs')
        for index, (observed, wanted) in enumerate(zip(actual, expected)):
            check_subset(observed, wanted, f'{label}[{index}]')
    elif isinstance(expected, float):
        if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f'{label}: {actual!r} != {expected!r}')
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f'{label}: {actual!r} != {expected!r}')


def replay(bundle, output_root, scripts):
    root = output_root / datetime.now(timezone.utc).strftime('compare-replay-%Y%m%dT%H%M%SZ')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root/'inputs'
    inputs.mkdir()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive members')
        index = json.loads(archive.read('EXPORT.json'))
        indexed = [entry['path'] for entry in index['files']]
        if len(indexed) != len(set(indexed)) or set(indexed) != set(names)-{'EXPORT.json'}:
            raise ValueError('archive index coverage mismatch')
        for entry in index['files']:
            data = archive.read(entry['path'])
            if hashlib.sha256(data).hexdigest() != entry['exported_sha256']:
                raise ValueError('archive digest mismatch: '+entry['path'])
            target = safe_path(inputs, entry['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
    sys.path.insert(0, str(scripts.resolve()))
    from compare_tensors import compare
    from compare_distributions import compare as compare_distributions
    from compare_target_scores import compare as compare_target_scores
    from validate_artifacts import validate
    operations = {'compare_tensors': compare, 'validate': validate,
                  'compare_distributions': compare_distributions,
                  'compare_target_scores': compare_target_scores}
    cases = json.loads((inputs/'REPLAY.json').read_text(encoding='utf-8'))['cases']
    if not cases:
        raise ValueError('replay cases required')
    names = [case['name'] for case in cases]
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise ValueError('unique nonempty replay case names required')
    results = []
    for index, case in enumerate(cases):
        paths = {key+'_path': safe_path(inputs, path) for key, path in case['inputs'].items()}
        report = operations[case['operation']](**paths)
        filename = f'case-{index:02d}.json'
        write(root/filename, report)
        result = dict(name=case['name'], report=filename, assertion='matched')
        try:
            check_subset(report, case['expected'])
        except AssertionError as exc:
            result.update(assertion='failed', reason=str(exc))
        results.append(result)
    result = dict(bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                  scripts=str(scripts.resolve()), cases=results,
                  matched=all(r['assertion']=='matched' for r in results))
    write(root/'replay.json', result)
    return root, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--scripts', type=Path, default=Path(__file__).resolve().parents[3]/'skills/precision-skill/scripts')
    args = parser.parse_args()
    try:
        root, result = replay(args.bundle, args.output_root, args.scripts)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(dict(run=str(root.resolve()), matched=result['matched'], cases=len(result['cases']))))
    return 0 if result['matched'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
