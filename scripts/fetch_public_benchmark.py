#!/usr/bin/env python3
"""Fetch only the 12 real OPEN100 PNG/GraphML pairs from PID2Graph's ZIP64.

Uses bounded HTTPS Range reads; never downloads the synthetic 9.3 GB archive.
Data: Stürmer et al., PID2Graph v1, Zenodo 14803338, CC BY-SA 4.0.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import zlib

import httpx

RECORD = 'https://zenodo.org/records/14803338'
URL = 'https://zenodo.org/api/records/14803338/files/PID2Graph.zip/content'
SIZE = 9303633645


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.client = httpx.Client(timeout=120, follow_redirects=True)
        self.downloaded = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else SIZE) + offset
        if not 0 <= self.pos <= SIZE:
            raise ValueError('invalid archive offset')
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = SIZE-self.pos if size < 0 else min(size, SIZE-self.pos)
        if not size:
            return b''
        if size > 16_000_000:
            raise ValueError('unexpectedly large archive member; refusing full download')
        start, end = self.pos, self.pos+size-1
        cache = Path('.benchmark-data/.ranges') / f'{start}-{end}.bin'
        if cache.exists() and cache.stat().st_size == size:
            self.pos += size
            return cache.read_bytes()
        for attempt in range(4):
            response = self.client.get(URL, headers={'Range': f'bytes={start}-{end}'})
            if response.status_code in (429, 503):
                time.sleep(min(60, int(response.headers.get('Retry-After', '5'))))
                continue
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}' or len(response.content) != size:
                raise ValueError('server did not honor the exact bounded byte range')
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.content)
            self.pos += size
            self.downloaded += size
            return response.content
        raise RuntimeError('dataset server rate limited all attempts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('.benchmark-data/open100'))
    p.add_argument('--list', action='store_true')
    args = p.parse_args()
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        members = sorted(f for f in archive.namelist() if '/Complete/PID2Graph OPEN100/' in '/'+f and f.lower().endswith(('.png','.graphml')))
        if len(members) != 24:
            raise ValueError(f'expected 12 pairs, found {len(members)} members: {members}')
        if args.list:
            print(json.dumps(members, indent=2))
            return
        args.output.mkdir(parents=True, exist_ok=True)
        lock = Path(__file__).resolve().parents[1]/'evaluation/data_manifest.json'
        expected = {f['file']:f['sha256'] for f in json.loads(lock.read_text())['files']} if lock.exists() else {}
        files = []
        for name in members:
            info = archive.getinfo(name)
            path = args.output / Path(name).name
            # ZipFile verifies each member's CRC32 on read; retain SHA256 too.
            data = path.read_bytes() if path.exists() else b''
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                data = archive.read(name)
            digest = hashlib.sha256(data).hexdigest()
            if expected and expected.get(path.name) != digest:
                raise ValueError(f'Pinned SHA256 mismatch: {path.name}')
            path.write_bytes(data)
            files.append({'archive_member':name,'file':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'crc32':f'{info.CRC:08x}'})
            print(path.name, len(data), flush=True)
        manifest = {'dataset':'PID2Graph v1 / Complete / PID2Graph OPEN100','source':RECORD,'archive_url':URL,'archive_bytes':SIZE,'license':'CC-BY-SA-4.0','authors':['Jan Marius Stürmer','Marius Graumann','Tobias Koch'],'drawing_creator':'Energy Impact Center / OPEN100','files':files}
        (args.output/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Downloaded {remote.downloaded:,} bytes using HTTPS ranges')

if __name__ == '__main__':
    main()
