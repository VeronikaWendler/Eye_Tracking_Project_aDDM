#!/usr/bin/env python3
"""Deterministic 5-fold, value-pair-grouped CV; no model fitting.

Identical (V_E, V_S) pairs across subjects/repetitions receive the SAME fold.
This is conservative grouping by fitted-value pair, not a claim of physical
stimulus identity. Fail closed when participant coverage/balance is inadequate.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

PRED=['AttentionW_SE','InattentionW_SE','AttentionContrast_SE','InattentionContrast_SE']
KEY=['sub_id','phase','trial']

def make(inp, out, study, k=5, seed=20261023):
    if k != 5: raise ValueError('This analysis plan fixes k=5; change only with documented amendment')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    src=Path(inp); raw=src.read_bytes(); source_hash=hashlib.sha256(raw).hexdigest()
    src_columns=list(pd.read_csv(src,nrows=0).columns)
    df=pd.read_csv(src,low_memory=False)
    req=KEY+['subj_idx','response','rt','V_E','V_S','PropDwell_E','PropDwell_S']+PRED
    missing=sorted(set(req)-set(df.columns))
    if missing: raise ValueError('Missing '+str(missing))
    if df[req].isnull().any().any(): raise ValueError('Missing required fitted input values')
    if df.duplicated(KEY).any(): raise ValueError('Duplicate unique-trial keys')
    if set(df.phase.astype(str).str.strip()) != {'ES'}: raise ValueError('Not an ES-only input')
    if not df.response.isin([0,1]).all() or not (df.rt>0).all(): raise ValueError('Response or RT invalid')
    if not np.allclose(df.PropDwell_E+df.PropDwell_S,1,atol=.002): raise ValueError('Gaze proportions invalid')
    if df.subj_idx.nunique() not in (20,21): raise ValueError('Expected 20/21 participants, not '+str(df.subj_idx.nunique()))
    # Coarser-than-physical-identity grouping is intentionally conservative.
    # Must match saved three-decimal precision. No outcomes used in grouping.
    ge=pd.to_numeric(df.V_E).round(3).map(lambda v:f'{v:.3f}')
    gs=pd.to_numeric(df.V_S).round(3).map(lambda v:f'{v:.3f}')
    df['cv_pair']=ge+'|'+gs
    ids=sorted(df.subj_idx.unique()); groups=df.cv_pair.unique().tolist()
    if len(groups)<k: raise ValueError('Insufficient unique value pairs')
    rng=np.random.RandomState(seed); perm=rng.permutation(len(groups))
    # Greedy multidimensional balance across individuals and total fold sizes;
    # never splits a value-pair group. Deterministic seeded tie breaking.
    subjects=pd.Index(ids)
    cnt=df.groupby(['cv_pair','subj_idx']).size().unstack(fill_value=0).reindex(index=groups,columns=subjects,fill_value=0)
    counts=cnt.to_numpy(float)
    target=counts.sum(axis=0)/k
    order=sorted(perm,key=lambda i:-counts[i].sum())
    loads=np.zeros((k,len(ids)))
    assignment={}
    for i in order:
        costs=[]
        for fold in range(k):
            candidate=loads.copy(); candidate[fold]+=counts[i]
            costs.append(np.sum((candidate-target)**2/np.maximum(target,1))+ 0.01*np.var(candidate.sum(axis=1)))
        best=min(costs); possibilities=[f for f,c in enumerate(costs) if np.isclose(c,best)]
        f=int(rng.choice(possibilities)); loads[f]+=counts[i];assignment[groups[i]]=f
    df['cv_fold']=df.cv_pair.map(assignment).astype(int)
    audit=[]
    for f in range(k):
        tst=df.cv_fold.eq(f);trn=~tst
        if df.loc[tst,'cv_pair'].isin(df.loc[trn,'cv_pair']).any(): raise AssertionError('Group leakage')
        for sid,g in df.groupby('subj_idx'):
            ntest=int((g.cv_fold==f).sum()); ntrain=len(g)-ntest
            if ntest<5 or ntrain<10 or ntest/len(g)<.05 or ntest/len(g)>.45:
                raise ValueError('Poor participant fold balance; study/fold/subj/test/total='+str((study,f,sid,ntest,len(g)))+'; revise grouping before fitting')
            audit.append(dict(study=study,fold=f,subj_idx=int(sid),n_train=ntrain,n_test=ntest,test_fraction=ntest/len(g)))
        if np.linalg.matrix_rank(df.loc[trn,PRED].to_numpy(float))!=4: raise ValueError('Training predictor rank deficiency')
        if df.loc[trn,'subj_idx'].nunique()!=len(ids): raise ValueError('Missing training participant')
        (out/f'fold_{f}').mkdir(exist_ok=True)
        df.loc[trn,src_columns].to_csv(out/f'fold_{f}'/'train.csv',index=False,float_format='%.3f')
        df.loc[tst,src_columns].to_csv(out/f'fold_{f}'/'test.csv',index=False,float_format='%.3f')
    pd.DataFrame(audit).to_csv(out/'fold_balance.csv',index=False)
    df[KEY+['subj_idx','cv_pair','cv_fold']].to_csv(out/'fold_assignments.csv',index=False)
    metadata={'study':study,'folds':k,'seed':seed,'source':str(src),'source_sha256':source_hash,
              'grouping':'3-decimal ordered (V_E,V_S) pairs, globally across subjects; repetitions never split',
              'total_rows':len(df),'subjects':len(ids),'value_pairs':len(groups),
              'response_coordinate':'S upper (1), E lower (0)',
              'folds_rows':df.cv_fold.value_counts().sort_index().to_dict()}
    (out/'fold_manifest.json').write_text(json.dumps(metadata,indent=2,default=int))
    print(json.dumps(metadata,indent=2,default=int),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--study',choices=['study1','study2'],required=True)
    a=p.parse_args()
    make(a.input,a.out,a.study)
