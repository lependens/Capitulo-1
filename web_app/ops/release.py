"""Release helpers for the existing LAN Compose project. No scientific writes.

snapshot saves current image IDs and environment before any build.
switch verifies a new derivative, changes WEB_DATA_DIR and recreates web only.
rollback uses saved images and environment, without rebuilding dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['snapshot','switch','rollback'])
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--deployed-sha', help='Observed production checkout SHA before pulling new source')
    a = p.parse_args()
    web = a.repo.resolve() / 'web_app'
    env = web / '.env'
    record = a.record.resolve()
    command = ['docker','compose','--project-name','siar-et0-web','--project-directory',str(web),'--env-file',str(env),'-f',str(web/'compose.yaml')]
    if a.action == 'snapshot':
        record.mkdir(parents=True, exist_ok=False, mode=0o700)
        shutil.copy2(env, record/'compose.env')
        (record/'compose.env').chmod(0o600)
        images = {}
        for service in ['frontend','backend']:
            image_id = run('docker','inspect','--format','{{.Image}}','siar-et0-web-'+service+'-1')
            tag = 'siar-et0-web-'+service+':rollback-'+record.name.lower()
            run('docker','image','tag',image_id,tag)
            images[service] = dict(id=image_id, tag=tag)
        data = dict(git_sha=a.deployed_sha or run('git','-C',str(a.repo),'rev-parse','HEAD'), images=images)
        (record/'rollback.json').write_text(json.dumps(data,indent=2))
        override = 'services:\n' + ''.join(f'  {s}:\n    image: {v["tag"]}\n' for s,v in images.items())
        (record/'rollback.yaml').write_text(override)
        print(json.dumps(dict(status='ROLLBACK_READY',record=str(record),**data)))
    elif a.action == 'switch':
        if not (record/'rollback.json').is_file():
            raise ValueError('A rollback snapshot is mandatory before switching')
        if not a.data:
            raise ValueError('--data is required')
        folder = a.data.resolve()
        info = json.loads((folder/'info.json').read_text())
        if info['schema_version'] != 'web-v0.3-explorer':
            raise ValueError('Wrong release derivative')
        for name,digest in info['artifacts'].items():
            if Path(name).name != name or hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Derivative integrity failure: '+name)
        value = env.read_text()
        if not re.search(r'^WEB_DATA_DIR=',value,re.M):
            raise ValueError('Missing WEB_DATA_DIR')
        value = re.sub(r'^WEB_DATA_DIR=.*$',lambda _: 'WEB_DATA_DIR='+str(folder),value,flags=re.M)
        env.write_text(value)
        print(run(*command,'up','-d','--no-build'))
        print(json.dumps(dict(status='SWITCHED',derived_sha256=info['derived_sha256'])))
    else:
        shutil.copy2(record/'compose.env',env)
        print(run(*command,'-f',str(record/'rollback.yaml'),'up','-d','--no-build'))
        print('ROLLBACK_APPLIED')


if __name__=='__main__':
    main()
