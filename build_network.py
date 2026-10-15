from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def build_network(r, out, example_index=0):
 """Write a standalone exploratory graph from a local feature table."""
 required = ['age','weight_kg','sbp','hr','spo2','creatinine','bun','sodium',
             'potassium','age_under18','diuretic_given','sex','hf_any','mv_pre6h',
             't0_after_icu_out','t0_after_death']
 missing = sorted(set(required) - set(r.columns))
 if missing:
  raise ValueError('Missing required columns: ' + ', '.join(missing))
 if len(r) < 2:
  raise ValueError('At least two patient rows are required.')
 if not 0 <= example_index < len(r):
  raise ValueError('--example-index must be a zero-based row index within the input.')
 for c in required:
  if c != 'sex':
   r = r.copy() if c == required[0] else r
   r[c] = pd.to_numeric(r[c], errors='raise')
   if np.isinf(r[c]).any():
    raise ValueError('Infinite values are not supported in ' + c)
 for c in ['age_under18','diuretic_given','hf_any','mv_pre6h','t0_after_icu_out','t0_after_death']:
  if not r[c].isin([0,1]).all():
   raise ValueError(c + ' must contain only 0 or 1, without missing values.')
 if not r.sex.isin(['F','M']).all():
  raise ValueError('This version supports sex codes F and M only.')
 r=r.reset_index(drop=True)
 out.mkdir(parents=True,exist_ok=True)
 cols=['age','weight_kg','sbp','hr','spo2','creatinine','bun','sodium','potassium']
 labels=['나이 (세)','체중 (kg)','수축기혈압 (mmHg)','심박수 (회/분)','SpO₂ (%)','Creatinine (mg/dL)','BUN (mg/dL)','나트륨 (mmol/L)','칼륨 (mmol/L)']
 v=r[cols].copy()
 for c in cols:
  if c+'_missing_outlier' in r:v.loc[r[c+'_missing_outlier'].ne(0),c]=np.nan
 valid=v.notna().to_numpy();counts=valid.astype(np.int16)@valid.astype(np.int16).T
 scale=v.quantile(.75)-v.quantile(.25)
 scale=scale.where(scale.gt(0),1.0).fillna(1.0)
 x=v.div(scale).to_numpy();sums=np.zeros(counts.shape)
 for j in range(len(cols)):
  a=x[:,j];sums+=np.nan_to_num(np.abs(a[:,None]-a[None,:]))
 distance=np.divide(sums,counts,out=np.full_like(sums,np.inf),where=counts>=5)+.3*(9-counts)/9
 agegroup=r.age_under18.to_numpy();distance[agegroup[:,None]!=agegroup[None,:]]=np.inf
 np.fill_diagonal(distance,np.inf)
 N=len(r)
 near=[]
 for n in range(N):
  ids=np.argsort(distance[n],kind='stable')[:10]
  near.append([[int(j),round(float(distance[n,j]),7),int(counts[n,j])] for j in ids if np.isfinite(distance[n,j])])
 def edges_for(k,t,minshared=5):
  edges={}
  for a,ns in enumerate(near):
   for b,d,c in ns[:k]:
    if d<=t and c>=minshared:edges[tuple(sorted((a,b)))]=(d,c)
  return edges
 edges=edges_for(5,.5)
 degree=np.zeros(N,dtype=int)
 for a,b in edges:degree[a]+=1;degree[b]+=1
 # Layout alone uses median-filled coordinates; these values never enter the distance.
 z=((v-v.median())/scale).fillna(0).to_numpy();u,s,_=np.linalg.svd(z,full_matrices=False)
 pos=u[:,:2]*s[:2];pos=np.clip(pos,-4,4)
 pos/=np.maximum(pos.std(axis=0),1e-5)
 rng=np.random.default_rng(20261015);pos+=rng.normal(0,.05,pos.shape)
 active=np.flatnonzero(degree>0);local={int(j):a for a,j in enumerate(active)}
 ea=np.array([local[a] for a,b in edges],dtype=int);eb=np.array([local[b] for a,b in edges],dtype=int)
 if len(active):
  p=pos[active].copy();p/=max(np.abs(p).max(),1)
  for it in range(90):
   d2=cdist(p,p,'sqeuclidean')+.0008
   np.fill_diagonal(d2,np.inf)
   w=1/d2
   rep=p*w.sum(axis=1)[:,None]-w@p
   diff=p[ea]-p[eb];length=np.linalg.norm(diff,axis=1)
   force=diff*(length/.09)[:,None]
   attr=np.zeros_like(p);np.add.at(attr,ea,-force);np.add.at(attr,eb,force)
   movement=rep*.0007+attr*.11-p*.03
   norm=np.linalg.norm(movement,axis=1);temp=.035*(1-it/100)
   p+=movement/np.maximum(norm[:,None],1e-9)*np.minimum(norm,temp)[:,None]
  p-=p.mean(axis=0);p/=max(np.abs(p).max(),1e-9);p[:,0]*=.78
  pos[active]=p
 isolates=np.flatnonzero(degree==0)
 for order,j in enumerate(isolates):pos[j]=[1.02+(order%5)*.045,-.9+(order//5)*.18]
 case=example_index
 nodes=[]
 for n,row in r.iterrows():
  raw=[None if pd.isna(row[c]) else round(float(row[c]),3) for c in cols]
  nodes.append({'id':int(n),'alias':f'P{n+1:04d}','x':round(float(pos[n,0]),6),'y':round(float(pos[n,1]),6),'values':raw,'valid':[bool(a) for a in valid[n]],'observed':int(valid[n].sum()),'treated':int(row.diuretic_given),'minor':int(row.age_under18),'sex':str(row.sex),'hf':int(row.hf_any),'mv':int(row.mv_pre6h),'timeIssue':bool(row.t0_after_icu_out or row.t0_after_death),'example':n==case})
 rows=[]
 for (a,b),(d,c) in sorted(edges.items()):rows.append({'patient_a':nodes[a]['alias'],'patient_b':nodes[b]['alias'],'distance':d,'shared_features':c})
 pd.DataFrame(rows,columns=['patient_a','patient_b','distance','shared_features']).to_csv(out/'patient_network_edges.csv',index=False,encoding='utf-8-sig')
 a=[a for a,b in edges]+[b for a,b in edges];b=[b for a,b in edges]+[a for a,b in edges]
 adj=coo_matrix((np.ones(len(a)),(a,b)),shape=(N,N)).tocsr()
 ncomp,components=connected_components(adj)
 metadata={'nodes':N,'edges':len(edges),'isolates':int((degree==0).sum()),'components':int(ncomp),'treated':int(r.diuretic_given.sum()),'insufficient_features':int((valid.sum(axis=1)<5).sum()),'example_alias':nodes[case]['alias'],'scale':{c:float(scale[c]) for c in cols}}
 (out/'network_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
 data={'nodes':nodes,'nearest':near,'labels':labels,'columns':cols,'meta':metadata,'example':case}
 template=(Path(__file__).parent/'templates'/'network.html').read_text(encoding='utf-8')
 (out/'patient_similarity_network.html').write_text(template.replace('__NETWORK_DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False)))
 return metadata


def main():
 import argparse
 parser = argparse.ArgumentParser(description='Build a local patient similarity network (v1.0.0).')
 parser.add_argument('--input', type=Path, required=True, help='Local raw feature CSV; never uploaded by this program.')
 parser.add_argument('--output-dir', type=Path, default=Path('output'))
 parser.add_argument('--example-index', type=int, default=0, help='Zero-based row highlighted as example patient A (default: 0).')
 args = parser.parse_args()
 try:
  data = pd.read_csv(args.input, encoding='utf-8-sig')
  summary = build_network(data, args.output_dir, args.example_index)
 except (ValueError, OSError) as exc:
  parser.exit(1, f'Build failed: {exc}\n')
 print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
 main()
