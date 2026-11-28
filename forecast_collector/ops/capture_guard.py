#!/usr/bin/python3
"""Fail closed when recent RAW integrity or local capacity is unsafe."""
import hashlib,os,re,sys
from pathlib import Path,PurePosixPath
ROOT=Path('/srv/siar-forecast/raw')
MIN_BYTES=2*1024**3
MIN_INODES=1000

def check(root=ROOT,min_bytes=MIN_BYTES,min_inodes=MIN_INODES):
 if root.is_symlink() or not root.is_dir():raise ValueError('raw_root_unavailable')
 st=os.statvfs(root)
 if st.f_bavail*st.f_frsize<min_bytes:raise ValueError('insufficient_free_space')
 if st.f_favail<min_inodes:raise ValueError('insufficient_free_inodes')
 tmp=root/'tmp';runs=root/'runs'
 if tmp.is_symlink() or not tmp.is_dir() or not os.access(tmp,os.W_OK|os.X_OK):raise ValueError('raw_staging_unavailable')
 if runs.is_symlink() or not runs.is_dir():raise ValueError('runs_unavailable')
 candidates=sorted(runs.glob('[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/run_*/run.json'))
 if not candidates:return 'no_published_runs'
 d=candidates[-1].parent
 if any(x.is_symlink() for x in (d,d.parent,d.parent.parent,d.parent.parent.parent)):raise ValueError('unsafe_run_path')
 sums=d/'SHA256SUMS'
 if sums.is_symlink() or not sums.is_file():raise ValueError('manifest_missing')
 count=0
 for line in sums.read_text(encoding='utf-8').splitlines():
  m=re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)',line)
  if not m:raise ValueError('manifest_invalid')
  rel=PurePosixPath(m.group(2))
  if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('unsafe_manifest_path')
  p=d.joinpath(*rel.parts)
  if not p.is_file() or p.is_symlink():raise ValueError('file_missing_or_symlink')
  p.resolve(strict=True).relative_to(d.resolve(strict=True))
  h=hashlib.sha256()
  with p.open('rb') as f:
   for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
  if h.hexdigest()!=m.group(1).lower():raise ValueError('hash_mismatch')
  count+=1
 if not count:raise ValueError('empty_manifest')
 return 'verified_last_run'
if __name__=='__main__':
 try:print('SIAR_CAPTURE_GUARD',check(),flush=True)
 except (OSError,ValueError) as e:print('SIAR_CAPTURE_GUARD ERROR',str(e),file=sys.stderr,flush=True);sys.exit(1)
