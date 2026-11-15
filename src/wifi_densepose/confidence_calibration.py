"""Independent scalar temperature fitting and subject-disjoint selective evaluation."""
from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .confidence import CHECKPOINTS, K4, K4_SHA, M3, M3_SHA, WindowID, probabilities

TARGETS = (1.,.9,.8,.7,.6)


def nll(logits, y, temperature):
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1)-scaled[np.arange(len(y)),y]))


def fit_temperature(logits,y):
    a,b=np.asarray(logits,dtype=np.float64),np.asarray(y)
    if (a.ndim!=2 or len(a)!=len(b) or not len(b) or not np.issubdtype(b.dtype,np.integer)
            or np.any(b<0) or np.any(b>=a.shape[1]) or not np.isfinite(a).all()):
        raise ValueError('Invalid fitting labels or logits')
    result=minimize_scalar(lambda t:nll(a,b,math.exp(t)),bounds=(math.log(.05),math.log(20.)),method='bounded',options={'xatol':1e-9})
    if not result.success: raise ValueError('Temperature optimization failed')
    return math.exp(result.x)


def classifier_stats(y,pred,k):
    cm=np.bincount(k*y+pred,minlength=k*k).reshape(k,k)
    diag=cm.diagonal(); support=cm.sum(1); pred_support=cm.sum(0)
    recall=np.divide(diag,support,out=np.zeros(k),where=support>0)
    f1=np.divide(2*diag,support+pred_support,out=np.zeros(k),where=(support+pred_support)>0)
    return {'accuracy':float(diag.sum()/len(y)) if len(y) else None,'macro_f1':float(f1.mean()) if len(y) else None,
            'recall':[float(z) if n else None for z,n in zip(recall,support)],'support':support.tolist()}


def calibration_metrics(logits,y,temp):
    scores=probabilities(logits,temp); pred=scores.argmax(1); conf=scores.max(1); correct=pred==y
    buckets=np.minimum((conf*15).astype(int),14)
    bins=[];ece=0.
    for b in range(15):
        mask=buckets==b; c=int(mask.sum())
        avg=float(conf[mask].mean()) if c else None;acc=float(correct[mask].mean()) if c else None
        if c: ece+=c/len(y)*abs(avg-acc)
        bins.append({'bin':b,'count':c,'confidence':avg,'accuracy':acc})
    return classifier_stats(y,pred,scores.shape[1]) | {
        'nll':nll(logits,y,temp),'brier':float(np.mean(np.sum((scores-np.eye(scores.shape[1])[y])**2,axis=1))),
        'ece':ece,'bins':bins,'n':len(y)}


def split_rows(rows, fit_subjects, eval_subjects):
    fit, evaluation = set(fit_subjects),set(eval_subjects)
    if not fit or not evaluation or fit&evaluation:
        raise ValueError('Calibration and evaluation subjects must be disjoint')
    seen=set();a=[];b=[]
    for row in rows:
        identity=row['identity'];subject=identity['subject']
        key=(subject,identity['sequence'],tuple(identity['frames']))
        if key in seen: raise ValueError('Duplicate prediction interval')
        seen.add(key)
        if subject in fit:a.append(row)
        if subject in evaluation:b.append(row)
    if {r['identity']['subject'] for r in a}!=fit or {r['identity']['subject'] for r in b}!=evaluation:
        raise ValueError('Subject missing from prediction artifact')
    return a,b


def threshold_for_coverage(conf, coverage):
    if not len(conf) or not 0 < coverage <= 1:
        raise ValueError('Coverage must be positive and no greater than one')
    return 0. if coverage==1. else float(np.sort(conf)[-math.ceil(len(conf)*coverage)])


def selective(y, pred, conf, threshold, k):
    keep=conf>=threshold
    rejected=int((~keep).sum());accepted=int(keep.sum())
    class_support=np.bincount(y,minlength=k)
    hits=np.bincount(y[keep & (pred==y)],minlength=k)
    return classifier_stats(y[keep],pred[keep],k) | {
        'total':len(y),'accepted':accepted,'coverage':accepted/len(y) if len(y) else None,'rejected':rejected,
        'rejected_error_rate':float(np.mean(y[~keep]!=pred[~keep])) if rejected else None,
        'end_to_end_recall':[float(z/n) if n else None for z,n in zip(hits,class_support)]}


def assess(rows,fit_subjects,eval_subjects):
    a,b=split_rows(rows,fit_subjects,eval_subjects)
    result={'status':'EXPLORATORY_ONLY','fit_subjects':list(fit_subjects),'eval_subjects':list(eval_subjects),'heads':{}}
    for head, order, taxonomy in [('posture',K4,K4_SHA),('motion',M3,M3_SHA)]:
        def parse(entries):
            kept=[r for r in entries if r['labels'][head] is not None]
            if not kept or any(tuple(r[head]['classes'])!=order or r[head]['model_sha256']!=CHECKPOINTS[head][0]
                               or r[head]['taxonomy_sha256']!=taxonomy or
                               WindowID(**r[head]['identity'])!=WindowID(**r['identity']) for r in kept):
                raise ValueError('Missing labels or inconsistent model, class order, or interval')
            x=np.asarray([r[head]['logits'] for r in kept],dtype=np.float64)
            y=np.asarray([r['labels'][head] for r in kept])
            if x.shape!=(len(kept),len(order)) or np.any(y<0) or np.any(y>=len(order)) or not np.issubdtype(y.dtype,np.integer):
                raise ValueError('Wrong logit dimension or labels')
            return x,y,np.array([r['identity']['subject'] for r in kept])
        x,y,_=parse(a);z,t,s= parse(b)
        temp=fit_temperature(x,y)
        coverage={}
        for variant,temperature in [('uncalibrated',1.),('temperature_exploratory',temp)]:
            fitconf=probabilities(x,temperature).max(1); score=probabilities(z,temperature)
            conf,pred=score.max(1),score.argmax(1)
            coverage[variant]=[]
            for target in TARGETS:
                cut=threshold_for_coverage(fitconf,target)
                coverage[variant].append({'target':target,'threshold_fit_only':cut,'fit_coverage':float(np.mean(fitconf>=cut)),
                    'evaluation':selective(t,pred,conf,cut,len(order)),
                    'subjects':{name:selective(t[s==name],pred[s==name],conf[s==name],cut,len(order)) for name in eval_subjects}})
        result['heads'][head]={'classes':list(order),'temperature_exploratory':temp,'fit_count':len(y),'eval_count':len(t),
             'fit_excluded':len(a)-len(y),'eval_excluded':len(b)-len(t),
             'before':calibration_metrics(z,t,1.),'after':calibration_metrics(z,t,temp),'coverage':coverage}
    return result


def reliability_svg(head,result):
    title=escape(f'{head} reliability · exploratory · n={result["eval_count"]}')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="505" viewBox="0 0 720 505">',
         '<rect width="720" height="505" fill="white"/>','<g font-family="Arial,sans-serif" fill="#173148">',
         f'<text x="42" y="34" font-size="22">{title}</text>',
         '<text x="42" y="58" font-size="13">15 equal bins · marker size reflects eval count · same held-out subjects</text>']
    x=lambda v:78+430*v
    y=lambda v:424-327*v
    for v in np.linspace(0,1,6):
        svg.extend([f'<path d="M{x(0)},{y(v)} H{x(1)} M{x(v)},{y(0)} V{y(1)}" fill="none" stroke="#e0e7ef"/>',
                    f'<text x="{x(v)}" y="445" font-size="12" text-anchor="middle">{v:.1f}</text>',
                    f'<text x="67" y="{y(v)+4}" font-size="12" text-anchor="end">{v:.1f}</text>'])
    svg.append(f'<path d="M{x(0)},{y(0)} L{x(1)},{y(1)}" fill="none" stroke="#8295a6" stroke-dasharray="5 4"/>')
    for i,(key,label,color) in enumerate([('before','Uncalibrated','#165f91'),('after','Temperature (exploratory)','#be552b')]):
        bins=[b for b in result[key]['bins'] if b['count']]
        points=' '.join(f'{x(b["confidence"]):.2f},{y(b["accuracy"]):.2f}' for b in bins)
        svg.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        for b in bins:
            r=3+10*math.sqrt(b['count']/result['eval_count'])
            svg.append(f'<circle cx="{x(b["confidence"]):.2f}" cy="{y(b["accuracy"]):.2f}" r="{r:.2f}" fill="{color}" opacity=".8"/>')
        svg.append(f'<text x="525" y="{120+24*i}" font-size="12" fill="{color}">{label}</text>')
    svg.extend(['<text x="293" y="479" text-anchor="middle" font-size="13">Mean top-1 confidence</text>',
                '<text x="18" y="257" transform="rotate(-90 18 257)" text-anchor="middle" font-size="13">Accuracy</text>','</g></svg>'])
    return '\n'.join(svg)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--fit-subjects',nargs='+',required=True);p.add_argument('--eval-subjects',nargs='+',required=True)
    a=p.parse_args()
    with a.predictions.open(encoding='utf-8-sig') as f:rows=[json.loads(line) for line in f if line.strip()]
    result=assess(rows,a.fit_subjects,a.eval_subjects)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'calibration_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,detail in result['heads'].items():
        (a.output/f'{name}_reliability.svg').write_text(reliability_svg(name,detail),encoding='utf-8')


if __name__=='__main__':main()
