#!/usr/bin/env python3
"""Service-only manifests compatible with the existing Haesoo boot gate."""
import argparse
import json
import re
from pathlib import Path

NAME='lineage'

def render(image,domain):
    if not re.fullmatch(r'lineage\.haesoo\.(dev|stream)',domain):raise ValueError('Use the dedicated lineage host.')
    if not image or any(c.isspace() for c in image):raise ValueError('Invalid image reference')
    labels={'app':NAME,'haesoo.dev/managed':'true','argocd.argoproj.io/instance':'lineage'}
    def obj(kind,api='v1',**kwargs):return {'apiVersion':api,'kind':kind,'metadata':{'name':NAME,'namespace':'apps','labels':labels},**kwargs}
    deployment=obj('Deployment','apps/v1',spec={
      'replicas':0,'revisionHistoryLimit':3,'strategy':{'type':'Recreate'},'selector':{'matchLabels':{'app':NAME}},
      'template':{'metadata':{'labels':labels},'spec':{
        'automountServiceAccountToken':False,'securityContext':{'runAsNonRoot':True,'runAsUser':1000,'runAsGroup':1000,'fsGroup':1000,'fsGroupChangePolicy':'OnRootMismatch'},
        'containers':[{'name':'app','image':image,'imagePullPolicy':'Never',
          'command':['/bin/sh','/platform/boot-gate.sh'],
          'args':['python','-m','uvicorn','backend.main:app','--host','0.0.0.0','--port','8000','--workers','1'],
          'ports':[{'containerPort':8000}],
          'env':[{'name':'DATA_DIR','value':'/data'},{'name':'PUBLIC_ORIGIN','value':'https://'+domain},
                 {'name':'COOKIE_SECURE','value':'1'},{'name':'APP_PASSWORD_HASH','valueFrom':{'secretKeyRef':{'name':'lineage-auth','key':'password-hash'}}}],
          'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
          'resources':{'requests':{'cpu':'100m','memory':'256Mi'},'limits':{'cpu':'2','memory':'1Gi'}},
          'readinessProbe':{'httpGet':{'path':'/healthz','port':8000},'periodSeconds':5,'timeoutSeconds':3},
          'volumeMounts':[{'name':'data','mountPath':'/data'},{'name':'tmp','mountPath':'/tmp'},
                          {'name':'gate','mountPath':'/platform','readOnly':True},{'name':'approval','mountPath':'/approval','readOnly':True}]}],
        'volumes':[{'name':'data','persistentVolumeClaim':{'claimName':'lineage-data'}},{'name':'tmp','emptyDir':{}},
                   {'name':'gate','configMap':{'name':'haesoo-boot-gate'}},
                   {'name':'approval','configMap':{'name':'haesoo-runtime','optional':True,'items':[{'key':NAME,'path':'boot-id'}]}}]}}})
    deployment['metadata']['annotations']={'haesoo.dev/title':'LINEAGE 도면 검토','haesoo.dev/domain':domain,'haesoo.dev/boot-gate':'v1'}
    pvc={'apiVersion':'v1','kind':'PersistentVolumeClaim','metadata':{'name':'lineage-data','namespace':'apps','labels':{'app':NAME}},'spec':{'accessModes':['ReadWriteOnce'],'storageClassName':'local-path','resources':{'requests':{'storage':'10Gi'}}}}
    service=obj('Service',spec={'selector':{'app':NAME},'ports':[{'port':8000,'targetPort':8000}],'type':'ClusterIP'})
    route=obj('HTTPRoute','gateway.networking.k8s.io/v1',spec={'parentRefs':[{'name':'home','namespace':'edge','sectionName':'web'}],'hostnames':[domain],'rules':[{'backendRefs':[{'name':NAME,'port':8000}]}]})
    return {'apiVersion':'v1','kind':'List','items':[pvc,deployment,service,route]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--domain',default='lineage.haesoo.dev');p.add_argument('--output',default='deploy/rendered.json');a=p.parse_args()
    target=Path(a.output);target.parent.mkdir(exist_ok=True,parents=True);target.write_text(json.dumps(render(a.image,a.domain),ensure_ascii=False,indent=2)+'\n');print(target)
