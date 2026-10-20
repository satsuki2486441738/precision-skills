"""Verify/export evidence ZIP volumes and reassemble unchanged binary captures.

Uses only the standard library; never deserializes tensor/pickle contents.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', required=True)
    ap.add_argument('--volumes', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    index = json.loads(Path(a.index).read_text())
    volume_root, out = Path(a.volumes), Path(a.output).resolve()
    if os.name == 'nt' and not str(out).startswith('\\\\?\\'):
        out = Path('\\\\?\\' + str(out))
    out.mkdir(parents=True, exist_ok=False)
    grouped, seen = {}, set()
    for volume in index['volumes']:
        path = volume_root / volume['filename']
        raw = path.read_bytes()
        assert len(raw) == volume['bytes'] and sha(raw) == volume['sha256'], path
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read('volume-manifest.json'))
            assert manifest['entries'] == volume['entries']
            assert set(z.namelist()) == {'volume-manifest.json'} | {e['archive_path'] for e in volume['entries']}
            for e in volume['entries']:
                relative = PurePosixPath(e['archive_path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                dest = (out / Path(*relative.parts)).resolve()
                assert dest.is_relative_to(out) and e['archive_path'] not in seen
                seen.add(e['archive_path'])
                data = z.read(e['archive_path'])
                assert len(data) == e['chunk_bytes'] and sha(data) == e['chunk_sha256']
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as f: f.write(data)
                grouped.setdefault(e['logical_path'], []).append(e)
    for name, entries in grouped.items():
        first = entries[0]
        target = (out / Path(*PurePosixPath(name).parts)).resolve()
        assert target.is_relative_to(out)
        if first['chunked']:
            offset = 0
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                for entry in sorted(entries, key=lambda e: e['offset']):
                    assert entry['offset'] == offset
                    data = (out / Path(*PurePosixPath(entry['archive_path']).parts)).read_bytes()
                    f.write(data)
                    offset += len(data)
            assert offset == first['archived_file_bytes']
        assert target.stat().st_size == first['archived_file_bytes']
        assert sha(target.read_bytes()) == first['archived_file_sha256']
        if first['tensor_bytes_unchanged']: assert sha(target.read_bytes()) == first['source_sha256']
    report = {'status': 'all_volume_and_file_hashes_verified', 'volumes': len(index['volumes']), 'logical_files': len(grouped), 'canonical_cases': index['canonical_case_count'], 'historical_completed_cases': index['historical_completed_case_count'], 'tensors_deserialized': False}
    (out / 'reassembly-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
