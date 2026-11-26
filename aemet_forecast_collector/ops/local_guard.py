import importlib.util, os, pathlib, shutil, sys
RAW=pathlib.Path('/srv/aemet-forecast/raw')
MIN_FREE=1024**3

def check(root, free_bytes, verifier):
    if free_bytes < MIN_FREE:
        raise RuntimeError('insufficient free space')
    for p in (root, root/'runs', root/'tmp'):
        if p.is_symlink() or not p.is_dir() or not os.access(p, os.W_OK|os.X_OK):
            raise RuntimeError('unsafe or unwritable data directory')
    runs=sorted((root/'runs').glob('*/*/*/run_*'))
    for run in runs:
        verifier(run)
    return len(runs)

def main():
    spec=importlib.util.spec_from_file_location('aemet_ops','/opt/aemet-forecast/current/ops.py')
    ops=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    n=check(RAW,shutil.disk_usage(RAW).free,ops.verify)
    print('AEMET_LOCAL_GUARD_PASS runs=%d free_bytes=%d' % (n,shutil.disk_usage(RAW).free))

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('AEMET_LOCAL_GUARD_STOP reason='+type(e).__name__,file=sys.stderr)
        sys.exit(1)
