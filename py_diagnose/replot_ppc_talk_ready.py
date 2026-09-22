#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd

OE,PE,OS,PS='deepskyblue','steelblue','darkorchid','indigo'
DW=['E>>S','E>S','S~E','S>E','S>>E']; RT=['1','2','3','4','5']
plt.rcParams.update({'font.size':21,'axes.titlesize':28,'axes.labelsize':24,
'xtick.labelsize':21,'ytick.labelsize':21,'legend.fontsize':18,
'axes.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42})

def clean(a):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.tick_params(axis='both',labelsize=21,width=1.3); a.grid(False)
def err(m,l,h):
    m,l,h=[np.asarray(x,float) for x in (m,l,h)]; return np.vstack([m-l,h-m])
def save(f,o,n):
    o.mkdir(parents=True,exist_ok=True)
    f.savefig(o/(n+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    f.savefig(o/(n+'.pdf'),bbox_inches='tight',facecolor='white'); plt.close(f)

def pairplot(d,study,title,ylabel,out,name,pct=False):
    order=[('E','Observed'),('E','Posterior predictive'),('S','Observed'),('S','Posterior predictive')]
    labs=['E\nObserved','E\nPredicted','S\nObserved','S\nPredicted']; cols=[OE,PE,OS,PS]
    x=np.array([0,.95,2.35,3.30]); m=[]; lo=[]; hi=[]
    for c,s in order:
        z=d[(d.choice==c)&(d.source==s)]
        if len(z)!=1: raise ValueError(f'Expected one row for {c}/{s}, got {len(z)}')
        m.append(float(z['mean'].iloc[0])); lo.append(float(z['low'].iloc[0])); hi.append(float(z['high'].iloc[0]))
    m,lo,hi=np.array(m),np.array(lo),np.array(hi)
    if pct: m*=100; lo*=100; hi*=100
    f,a=plt.subplots(figsize=(10.5,7.3)); a.bar(x,m,.78,color=cols,edgecolor='black',linewidth=1.4)
    a.errorbar(x,m,yerr=err(m,lo,hi),fmt='none',ecolor='black',elinewidth=1.8,capsize=6)
    a.set_xticks(x); a.set_xticklabels(labs); a.set_ylabel(ylabel)
    if pct:a.set_ylim(0,100)
    a.set_title(f'{study}: {title}\nObserved vs posterior predictive',pad=16); clean(a); save(f,out,name)

def dwell(obs,sim,study,out):
    o=obs.set_index('bin').reindex(DW)
    p=(sim.groupby('dwell_prop_quintile',observed=False)['p_choose_s']
       .agg(mean='mean',low=lambda s:s.quantile(.025),high=lambda s:s.quantile(.975)).reindex(DW))
    x=np.arange(5); om=100*o['mean'].to_numpy(float); ol=100*o['boot_ci_low'].to_numpy(float); oh=100*o['boot_ci_high'].to_numpy(float)
    pm=100*p['mean'].to_numpy(float); pl=100*p['low'].to_numpy(float); ph=100*p['high'].to_numpy(float)
    f,a=plt.subplots(figsize=(11.5,7.8)); a.bar(x,om,.70,color=OS,edgecolor='black',alpha=.82,linewidth=1.4)
    a.errorbar(x,om,yerr=err(om,ol,oh),fmt='none',ecolor='black',elinewidth=1.8,capsize=6)
    a.fill_between(x,pl,ph,color=PS,alpha=.15); a.plot(x,pm,color=PS,lw=3.5,marker='o',ms=9)
    a.set_xticks(x); a.set_xticklabels(DW); a.set_ylim(0,100); a.set_ylabel('P(choose S) (%)')
    a.set_xlabel('Proportional dwell advantage (S − E)'); a.set_title(f'{study}: Choice by proportional dwell quintile',pad=16)
    a.legend(handles=[Patch(facecolor=OS,edgecolor='black',alpha=.82,label='Observed'),Line2D([0],[0],color=PS,marker='o',lw=3.5,label='Posterior predictive'),Patch(facecolor=PS,alpha=.15,label='Predictive 95% interval')],frameon=False,loc='upper left')
    clean(a); save(f,out,'dwell_prop_quintile_talk_ready')

def rtq(d,study,out):
    d=d.copy(); d['bin']=d['bin'].astype(str); x=np.arange(5); w=.34; f,a=plt.subplots(figsize=(11.8,7.8))
    for c,off,oc,pc in [('E',-w/2,OE,PE),('S',w/2,OS,PS)]:
        z=d[d.choice==c].set_index('bin').reindex(RT); xp=x+off
        om=100*z.observed_mean.to_numpy(float); ol=100*z.observed_boot_ci_low.to_numpy(float); oh=100*z.observed_boot_ci_high.to_numpy(float)
        pm=100*z.ppc_mean.to_numpy(float); pl=100*z.ppc_pi95_low.to_numpy(float); ph=100*z.ppc_pi95_high.to_numpy(float)
        a.bar(xp,om,w*.92,color=oc,edgecolor='black',alpha=.82,linewidth=1.3)
        a.errorbar(xp,om,yerr=err(om,ol,oh),fmt='none',ecolor='black',elinewidth=1.6,capsize=5)
        a.plot(xp,pm,color=pc,marker='o',ms=8,lw=3); a.errorbar(xp,pm,yerr=err(pm,pl,ph),fmt='none',ecolor=pc,elinewidth=1.8,capsize=5)
    a.axhline(50,color='.65',ls=':',lw=1.5); a.set_xticks(x); a.set_xticklabels(RT); a.set_ylim(0,100)
    a.set_xlabel('RT quintile (1 = fastest, 5 = slowest)'); a.set_ylabel('Choice probability (%)'); a.set_title(f'{study}: Choice by RT quintile',pad=16)
    a.legend(handles=[Patch(facecolor=OE,edgecolor='black',alpha=.82,label='Observed E'),Line2D([0],[0],color=PE,marker='o',lw=3,label='Predicted E'),Patch(facecolor=OS,edgecolor='black',alpha=.82,label='Observed S'),Line2D([0],[0],color=PS,marker='o',lw=3,label='Predicted S')],frameon=False,ncol=2,loc='upper right')
    clean(a); save(f,out,'rt_quintile_choice_talk_ready')

def main():
    q=argparse.ArgumentParser(); q.add_argument('--study',required=True,choices=['study1','study2']); q.add_argument('--ppc-dir',type=Path,required=True); q.add_argument('--out-dir',type=Path,required=True); a=q.parse_args()
    study='Study 1' if a.study=='study1' else 'Study 2'; t=a.ppc_dir/'tables'
    files=['overall_choice_es_group_plot.csv','rt_by_response_group_plot.csv','observed_dwell_prop_summary_identity.csv','simulated_dwell_prop_by_draw_identity.csv','choice_e_s_by_rt_quintile_ppc.csv']
    for n in files:
        if not (t/n).exists(): raise FileNotFoundError(t/n)
    print(f'Talk-ready PPC replot — {study}. No PPC simulation is rerun.')
    pairplot(pd.read_csv(t/files[0]),study,'Overall choice','Choice probability (%)',a.out_dir,'overall_choice_talk_ready',True)
    pairplot(pd.read_csv(t/files[1]),study,'RT conditional on choice','Participant median RT (s)',a.out_dir,'rt_conditional_choice_talk_ready')
    dwell(pd.read_csv(t/files[2]),pd.read_csv(t/files[3]),study,a.out_dir)
    rtq(pd.read_csv(t/files[4]),study,a.out_dir)
    print('Saved PNG + PDF plots to',a.out_dir)
if __name__=='__main__': main()
