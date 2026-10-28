#!/usr/bin/env python3
"""Fetch the predeclared, previously unused public synthetic confirmation set."""
import hashlib,json,struct,zipfile
from pathlib import Path
from fetch_public_benchmark import RemoteZip,RECORD,URL
ROOT=Path(__file__).resolve().parents[1]
plan=json.loads((ROOT/'evaluation/v3/confirmation-plan.json').read_text())
out=ROOT/'.benchmark-data/pid-confirmation';out.mkdir(exist_ok=True,parents=True)
files=[]
with zipfile.ZipFile(RemoteZip()) as archive:
    for png in plan['members']:
        for name in (png,png[:-4]+'.graphml'):
            path=out/Path(name).name
            if path.exists():data=path.read_bytes()
            else:
                with archive.open(name) as source:
                    chunks=[]
                    while chunk:=source.read(1024*1024):chunks.append(chunk)
                    data=b''.join(chunks)
                path.write_bytes(data)
            info=archive.getinfo(name)
            entry={'file':path.name,'archive_member':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'crc32':f'{info.CRC:08x}'}
            if path.suffix=='.png':entry['original_size']=list(struct.unpack('>II',data[16:24]));entry['within_production_pixel_limit']=entry['original_size'][0]*entry['original_size'][1]<=50_000_000
            files.append(entry);print(path.name,len(data),entry.get('original_size',''),flush=True)
manifest={'source':RECORD,'archive_url':URL,'license':'CC-BY-SA-4.0','drawing_origin':'Digitize-PID / Dataset-P&ID (synthetic)','selection_plan':plan,'files':files}
(ROOT/'evaluation/v3/confirmation-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
