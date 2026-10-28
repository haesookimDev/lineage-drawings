#!/usr/bin/env python3
"""Import one CI-built image and deploy only LINEAGE; requires existing k3s admin access.
No credentials or source files are transmitted to a third party by this helper.
"""
import argparse
import base64
import hashlib
import json
import os
import secrets
import subprocess
from pathlib import Path
from render import render

ROOT=Path(__file__).resolve().parents[1]

def kubectl(*args,body=None,check=True):
    command=['k3s','kubectl',*args]
    result=subprocess.run(command,input=json.dumps(body).encode() if body else None,capture_output=True,check=False)
    if check and result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[:1000])
    return result

def get(*args):return json.loads(kubectl(*args,'-o','json').stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--domain',default='lineage.haesoo.dev');p.add_argument('--start',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if len(a.sha)!=40 or any(c not in '0123456789abcdef' for c in a.sha):raise SystemExit('Provide the full tested app commit SHA.')
    if not a.artifact.is_file():raise SystemExit('Download the CI container-image artifact first.')
    image='docker.io/library/lineage-drawings:'+a.sha
    manifest=render(image,a.domain)
    if a.dry_run:
        print(json.dumps(manifest,ensure_ascii=False,indent=2));return
    # Preflight all existing dependencies before mutating the cluster.
    nodes=get('get','nodes')
    if len(nodes['items'])!=1:raise SystemExit('This image-import deployment supports the documented single-node k3s cluster only.')
    kubectl('get','namespace','apps');kubectl('-n','apps','get','configmap','haesoo-boot-gate');kubectl('-n','apps','get','configmap','haesoo-runtime');kubectl('-n','edge','get','gateway','home')
    existing=kubectl('-n','apps','get','deployment','lineage','-o','json',check=False)
    if existing.returncode==0:
        meta=json.loads(existing.stdout)['metadata']
        if meta.get('labels',{}).get('app')!='lineage' or meta.get('annotations',{}).get('haesoo.dev/boot-gate')!='v1':raise SystemExit('An unrelated lineage deployment already exists; refusing to replace it.')
    elif b'NotFound' not in existing.stderr:raise RuntimeError(existing.stderr.decode())
    for route in get('-n','apps','get','httproutes')['items']:
        if a.domain in route['spec'].get('hostnames',[]) and route['metadata']['name']!='lineage':raise SystemExit('Domain is already used by another service.')
    subprocess.run(['k3s','ctr','--namespace','k8s.io','images','import',str(a.artifact.resolve())],check=True)
    # Provision app password once. Preserve it on subsequent deployments.
    found=kubectl('-n','apps','get','secret','lineage-auth','-o','name',check=False)
    if found.returncode:
        if b'NotFound' not in found.stderr:raise RuntimeError(found.stderr.decode())
        password=secrets.token_urlsafe(24);salt=secrets.token_hex(16)
        hashed=salt+':'+hashlib.pbkdf2_hmac('sha256',password.encode(),salt.encode(),600000).hex()
        sec={'apiVersion':'v1','kind':'Secret','metadata':{'name':'lineage-auth','namespace':'apps'},'type':'Opaque','data':{'password-hash':base64.b64encode(hashed.encode()).decode()}}
        kubectl('apply','-f','-',body=sec)
        local=ROOT/'.secrets';local.mkdir(mode=0o700,exist_ok=True)
        target=local/'login.txt';fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write('URL: https://'+a.domain+'\nPassword: '+password+'\n')
        if os.getenv('SUDO_UID'):
            os.chown(local,int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']));os.chown(target,int(os.environ['SUDO_UID']),int(os.environ['SUDO_GID']))
        print('Initial login saved locally (not printed):',target)
    # Revoke only LINEAGE's approval and stop only LINEAGE before applying a new version.
    kubectl('-n','apps','patch','configmap','haesoo-runtime','--type=merge','--patch-file=/dev/stdin',body={'data':{'lineage':None}})
    if existing.returncode==0:
        kubectl('-n','apps','scale','deployment/lineage','--replicas=0')
        kubectl('-n','apps','wait','--for=delete','pods','-l','app=lineage','--timeout=180s')
    result=kubectl('apply','-f','-',body=manifest);print(result.stdout.decode())
    if a.start:
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        kubectl('-n','apps','patch','configmap','haesoo-runtime','--type=merge','--patch-file=/dev/stdin',body={'data':{'lineage':boot}})
        kubectl('-n','apps','scale','deployment/lineage','--replicas=1')
        kubectl('-n','apps','rollout','status','deployment/lineage','--timeout=300s')
        print('Running: https://'+a.domain)
    else:print('Deployed stopped. Start LINEAGE from the existing service portal.')

if __name__=='__main__':main()
