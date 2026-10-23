#!/usr/bin/env python3
"""Exact HDDM Wiener likelihood on held-out trials integrated over training-only posterior.

Fail closed on missing columns, invalid parameter transforms, nonfinite densities,
unequal trial coverage, or likelihood non-additivity. Uses SAME wfpt function as
HDDMRegressor. No approximation by PPC bins or posterior mean parameters.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import hddm
from scipy.special import logsumexp, expit

BASE=['AttentionW_SE','InattentionW_SE']
CONTRAST=BASE+['AttentionContrast_SE','InattentionContrast_SE']
KEY=['sub_id','phase','trial']

def trace_column(traces,subject,base):
    col=f'{base}_subj.{int(subject)}'
    if col not in traces.columns:
        raise KeyError(f'Cannot locate subject trace {col}; available relevant='+repr([c for c in traces.columns if base in c][:25]))
    x=traces[col].to_numpy(dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Invalid trace '+col)
    return x

def get_z(traces,subject):
    name=f'z_subj.{int(subject)}'
    tr=f'z_subj_trans.{int(subject)}'
    if name in traces.columns: x=traces[name].to_numpy(dtype=float)
    elif tr in traces.columns: x=expit(traces[tr].to_numpy(dtype=float))
    else: raise KeyError(f'Missing {name} or {tr}: inspect HDDM node naming')
    if not np.isfinite(x).all() or not ((x>0)&(x<1)).all(): raise ValueError('Invalid z posterior transform')
    return x

def one_logp(rt,v,a,z,t):
    # Exact same compiled callable/arguments as HDDMRegressor: reg_outcomes={'v'}
    # rt is SIGNED: response 0 (E) => negative, response 1 (S) => positive.
    return float(hddm.wfpt.wiener_like_multi(
        np.asarray([rt],dtype=float),np.asarray([v],dtype=float),0.,float(a),float(z),0.,float(t),0.,1e-4,{'v'},
        w_outlier=.1,p_outlier=.05))

def additivity_self_test():
    rt=np.asarray([.71,-.94,.86],float);v=np.asarray([.1,-.3,.2],float)
    joint=float(hddm.wfpt.wiener_like_multi(rt,v,0.,1.5,.42,0.,.2,0.,1e-4,{'v'},w_outlier=.1,p_outlier=.05))
    singles=sum(one_logp(r,dr,1.5,.42,.2) for r,dr in zip(rt,v))
    if not np.isfinite(joint) or not np.isclose(joint,singles,rtol=0,atol=2e-5):
        raise RuntimeError(f'Wiener pointwise likelihood validation FAILED: joint {joint}, sum {singles}')
    print('PASS: Wiener likelihood additivity',joint,singles,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--test',type=Path,required=True);p.add_argument('--train',type=Path,required=True)
 p.add_argument('--fit-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--model',choices=['basic','contrast'],required=True);p.add_argument('--study',choices=['study1','study2'],required=True)
 p.add_argument('--fold',type=int,required=True);p.add_argument('--chains',type=int,default=3)
 p.add_argument('--draws-per-chain',type=int,default=200)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if not (a.fit_dir/'fit_complete.flag').exists(): raise RuntimeError('Missing fit_complete.flag — refusing partially fitted chains')
 if not (a.fit_dir/'fit_audit.json').exists():raise RuntimeError('Missing fit audit')
 meta=json.loads((a.fit_dir/'fit_audit.json').read_text())
 if (meta['study'],int(meta['fold']),int(meta['chains']))!=(a.study,a.fold,a.chains):raise ValueError('Wrong fit audit/study/fold/chains')
 test=pd.read_csv(a.test,low_memory=False);train=pd.read_csv(a.train,low_memory=False)
 if test.duplicated(KEY).any() or train.duplicated(KEY).any():raise ValueError('Duplicate trial key')
 both=pd.concat([train[KEY].assign(dataset='train'),test[KEY].assign(dataset='test')])
 if both.duplicated(KEY).any():raise ValueError('Training-test trial leakage')
 if set(map(tuple,train[['V_E','V_S']].round(3).to_numpy())) & set(map(tuple,test[['V_E','V_S']].round(3).to_numpy())):
  raise ValueError('Value-pair group leakage')
 if not set(test.subj_idx.unique()).issubset(set(train.subj_idx.unique())):raise ValueError('Unseen participants in test')
 if not test.response.isin([0,1]).all() or not (test.rt>0).all():raise ValueError('Invalid test response/RT')
 pred=BASE if a.model=='basic' else CONTRAST
 if test[pred].isnull().any().any():raise ValueError('Missing test regressors')
 additivity_self_test()
 tag=f'aDDM_ES_IDENTITY_S_UPPER_{a.model.upper()}_Z_NO_INTERCEPT'
 n=len(test);all_logps=[];draw_audit=[]
 for chain in range(a.chains):
  path=a.fit_dir/f'{tag}_{chain}.hddm'
  if not path.exists():raise FileNotFoundError(path)
  print('Loading',path,flush=True)
  model=hddm.load(str(path));traces=model.get_traces()
  if a.draws_per_chain>len(traces):raise ValueError('Not enough posterior draws per chain')
  # deterministic evenly spaced draws from each chain, preserving JOINT row indexing.
  selected=np.linspace(0,len(traces)-1,a.draws_per_chain,dtype=int)
  if len(np.unique(selected))!=len(selected):raise ValueError('Repeated sampled trace positions')
  lp=np.empty((n,len(selected)),dtype=float)
  for subj,idx in test.groupby('subj_idx').groups.items():
   idx=np.asarray(list(idx),dtype=int)
   aa=trace_column(traces,subj,'a')[selected]
   tt=trace_column(traces,subj,'t')[selected]
   zz=get_z(traces,subj)[selected]
   betas=np.column_stack([trace_column(traces,subj,'v_'+x)[selected] for x in pred])
   design=test.iloc[idx][pred].to_numpy(float)
   drift=design@betas.T
   signed=test.iloc[idx]['rt'].to_numpy(float)*np.where(test.iloc[idx]['response'].to_numpy(int)==1,1.,-1.)
   for k in range(len(selected)):
    for j,row in enumerate(idx):lp[row,k]=one_logp(signed[j],drift[j,k],aa[k],zz[k],tt[k])
   print(f'chain={chain} subject={subj} test_trials={len(idx)} scored',flush=True)
  if not np.isfinite(lp).all():raise ValueError('Nonfinite log-likelihood; inspect values and boundary mixture')
  all_logps.append(lp)
  draw_audit.append({'chain':chain,'available_draws':len(traces),'selected_draws':len(selected),'indices_first':int(selected[0]),'indices_last':int(selected[-1])})
 logp=np.concatenate(all_logps,axis=1)
 # *Log of mean density*, NOT mean of log densities.
 lpd=logsumexp(logp,axis=1)-np.log(logp.shape[1])
 if not np.isfinite(lpd).all():raise ValueError('Nonfinite integrated log predictive density')
 cols=KEY+['subj_idx','response','rt']
 result=test[cols].copy();result.insert(0,'study',a.study);result.insert(1,'fold',a.fold);result.insert(2,'model',a.model)
 result['heldout_lpd']=lpd
 result.to_csv(a.out/'heldout_pointwise.csv',index=False,float_format='%.10g')
 diagnostics={'study':a.study,'fold':a.fold,'model':a.model,'n_test':n,'n_chains':a.chains,
  'posterior_draws_used':logp.shape[1],'elpd_sum':float(lpd.sum()),'mean_lpd':float(lpd.mean()),
  'draw_audit':draw_audit,'w_outlier':.1,'p_outlier':.05,'signed_rt':'E negative; S positive',
  'likelihood':'hddm.wfpt.wiener_like_multi pointwise; reg_outcomes={v}; no posterior leakage'}
 (a.out/'score_audit.json').write_text(json.dumps(diagnostics,indent=2))
 print(json.dumps(diagnostics,indent=2),flush=True)
 (a.out/'score_complete.flag').write_text('all trial densities finite; consistency checks passed\n')

if __name__=='__main__':main()
