"""AEMET operational entry point. Never prints subprocess output or secrets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

ETC = Path('/etc/aemet-forecast')
RAW = Path('/srv/aemet-forecast/raw')
RCLONE = '/opt/aemet-forecast/tools/rclone/1.75.2/rclone'
DEST = 'forecast-offsite-crypt:AEMET-forecast/raw/runs'


def gate():
    p = ETC / 'pilot.gate'
    st = p.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022:
        raise ValueError('unsafe gate')
    if p.read_text().strip() != 'SIAR-FORECAST-PILOT = RUNNING':
        raise ValueError('gate closed')


def credentials(name):
    return Path(os.environ['CREDENTIALS_DIRECTORY']) / name


def verify(run):
    """Require exact manifest coverage; reject symlinks and path traversal."""
    if run.is_symlink() or not run.is_dir():
        raise ValueError('invalid run')
    entries = {}
    for line in (run / 'SHA256SUMS').read_text(encoding='ascii').splitlines():
        digest, name = line.split('  ', 1)
        p = Path(name)
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or p.is_absolute() or '..' in p.parts or name in entries
                or '\\' in name or p.as_posix() != name):
            raise ValueError('invalid manifest')
        entries[name] = digest
    files = set()
    for p in run.rglob('*'):
        if p.is_symlink():
            raise ValueError('symlink in run')
        if p.is_file():
            name = p.relative_to(run).as_posix()
            if name == 'SHA256SUMS':
                continue
            files.add(name)
            if hashlib.sha256(p.read_bytes()).hexdigest() != entries.get(name):
                raise ValueError('checksum mismatch')
        elif not p.is_dir():
            raise ValueError('special file in run')
    if files != set(entries) or 'run.json' not in files:
        raise ValueError('incomplete manifest coverage')
    data = json.loads((run / 'run.json').read_text())
    if data.get('schema_version') != 'aemet-raw-run-v1' or data.get('status') not in {'complete', 'partial', 'failed'}:
        raise ValueError('invalid run metadata')
    if data.get('run_id') != run.name:
        raise ValueError('run identity mismatch')


def cloud(*args):
    # Capture output: remote errors may contain OAuth or configuration details.
    result = subprocess.run([RCLONE, '--config', str(Path('/var/lib/aemet-forecast/rclone/rclone.conf')),
                             '--ask-password=false', '--retries', '1',
                             '--low-level-retries', '1', *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('cloud operation failed; raw retained')


def reconcile():
    gate()
    failures = 0
    runs = sorted((RAW / 'runs').glob('*/*/*/run_*'))
    for run in runs:
        try:
            verify(run)
            remote = DEST + '/' + run.relative_to(RAW / 'runs').as_posix()
            cloud('copy', str(run), remote, '--immutable')
            cloud('check', str(run), remote, '--download', '--one-way')
            # Marker is independent of capture completeness (partial/failed are valid).
            marker = {'schema': 'aemet-offsite-complete-v1',
                      'manifest_sha256': hashlib.sha256((run / 'SHA256SUMS').read_bytes()).hexdigest()}
            expected = (json.dumps(marker, sort_keys=True) + '\n').encode('utf-8')
            base = [RCLONE, '--config',
                    '/var/lib/aemet-forecast/rclone/rclone.conf',
                    '--ask-password=false', '--retries', '1',
                    '--low-level-retries', '1']
            listing = subprocess.run(
                base + ['lsf', remote, '--files-only', '--include', '/COMPLETE'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if listing.returncode:
                raise RuntimeError('remote marker listing failed')
            names = listing.stdout.decode('utf-8').splitlines()
            if names == ['COMPLETE']:
                existing = subprocess.run(
                    base + ['cat', remote + '/COMPLETE'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                if existing.returncode or existing.stdout != expected:
                    raise RuntimeError('remote marker mismatch or unreadable')
            elif not names:
                td = tempfile.mkdtemp(prefix='backup-', dir=RAW / 'tmp')
                p = Path(td) / 'COMPLETE'
                with p.open('xb') as stream:
                    stream.write(expected)
                cloud('copy', td, remote, '--immutable')
                cloud('check', td, remote, '--download', '--one-way')
            else:
                raise RuntimeError('unexpected remote marker listing')
            print('backup verified:', run.name)
        except Exception:
            failures += 1
            print('backup pending:', run.name)
    return 1 if failures else 0


def capture():
    gate()
    config = ETC / 'config.toml'
    from aemet_forecast_collector.config import load_config
    c = load_config(config)
    if c.data_root != RAW:
        raise ValueError('wrong raw root')
    key = credentials('api_key').read_text().strip()
    if not key:
        raise ValueError('missing credential')
    os.environ['AEMET_API_KEY'] = key
    from aemet_forecast_collector.cli import main
    return main(['capture', '--config', str(config)])


def restore(run_relative):
    gate()
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}/run_[A-Za-z0-9_.-]+', run_relative or ''):
        raise ValueError('invalid restore run path')
    root = Path(tempfile.mkdtemp(prefix='restore-', dir=RAW / 'tmp'))
    run = root / run_relative.split('/')[-1]
    run.mkdir(mode=0o2770)
    remote = DEST + '/' + run_relative
    cloud('copy', remote, str(run), '--immutable', '--exclude', '/COMPLETE')
    cloud('copy', remote + '/COMPLETE', str(root), '--immutable')
    verify(run)
    marker = json.loads((root / 'COMPLETE').read_text())
    if (marker.get('schema') != 'aemet-offsite-complete-v1' or marker.get('manifest_sha256') !=
            hashlib.sha256((run / 'SHA256SUMS').read_bytes()).hexdigest()):
        raise ValueError('restore marker mismatch')
    print('restore verified in quarantine:', root.name)
    return 0


def preflight():
    # Offline only: no API requests, no rclone execution, no raw creation.
    from aemet_forecast_collector.config import load_config
    c = load_config(ETC / 'config.toml')
    if c.data_root != RAW:
        raise ValueError('wrong raw root')
    for p in [RAW, RAW / 'runs', RAW / 'tmp']:
        if p.is_symlink() or p.stat().st_mode & 0o7777 != 0o2770:
            raise ValueError('raw directory permissions')
    for p in [ETC / 'config.toml', ETC / 'api_key']:
        st = p.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o007:
            raise ValueError('unsafe configuration')
    print('offline runtime/storage/config preflight PASS')
    print('credential provisioned =', 'YES' if (ETC / 'api_key').stat().st_size else 'NO')
    print('gate file present =', 'YES' if (ETC / 'pilot.gate').exists() else 'NO')
    print('published local runs =', len(list((RAW / 'runs').glob('*/*/*/run_*'))))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight', 'capture', 'reconcile', 'restore'])
    parser.add_argument('--run')
    args = parser.parse_args()
    try:
        sys.exit(restore(args.run) if args.command == 'restore' else
                 {'preflight': preflight, 'capture': capture, 'reconcile': reconcile}[args.command]())
    except Exception as exc:
        print('AEMET operation refused/failed (' + type(exc).__name__ + '); data retained', file=sys.stderr)
        sys.exit(1)
