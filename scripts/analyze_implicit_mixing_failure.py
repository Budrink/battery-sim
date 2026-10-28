#!/usr/bin/env python3
"""Inspect the exact frozen CUDA operator on failure; no simulation or fitting."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning

import calibrate_upv_r123 as cal


def analyze(snapshot):
    rows=snapshot['rows'];n=len(rows)
    x=np.asarray([r['x'] for r in rows]);origin=np.asarray([r['origin'] for r in rows])
    recorded=np.asarray([r['r'] for r in rows])
    mass=np.asarray([r['mobile'] for r in rows]);thermal=np.asarray([r['thermal'] for r in rows])
    gas=np.asarray(snapshot['gas_edges'],dtype=float).reshape(-1,3)
    film=np.asarray(snapshot['film_edges'],dtype=float).reshape(-1,3)
    result=dict(gas_count=snapshot['gas_count'],solid_count=n-snapshot['gas_count'],
        step_s=snapshot['step_s'],iterations=snapshot['iterations'],fields={})
    names=('temperature','chemical','oxygen','vapor','latent')
    eps=np.finfo(float).eps
    for f,name in enumerate(names):
        cap=thermal if f==0 else mass
        tolerance=snapshot['temperature_tolerance_k']/1000. if f==0 else snapshot['tracer_relative_tolerance']
        edges=np.concatenate([gas,film]) if f==0 else gas
        a,b=edges[:,:2].astype(int).T;w=edges[:,2]
        # Long-double difference form keeps the tiny capacity term separate
        # from the large edge terms; assembled double diagonals can lose it.
        xx=x[:,f].astype(np.longdouble);cc=cap.astype(np.longdouble)
        residual=cc*(origin[:,f].astype(np.longdouble)-xx)
        flux=w.astype(np.longdouble)*(xx[a]-xx[b])
        np.add.at(residual,a,-flux);np.add.at(residual,b,flux)
        norm=np.abs(residual)/(cc*tolerance)
        magnitude=np.abs(cap*origin[:,f])+np.abs(cap*x[:,f]);terms=np.full(n,4.)
        edge_magnitude=w*(np.abs(x[a,f])+np.abs(x[b,f]))
        np.add.at(magnitude,a,edge_magnitude);np.add.at(magnitude,b,edge_magnitude)
        np.add.at(terms,a,6.);np.add.at(terms,b,6.)
        roundoff=eps*terms*magnitude
        candidates=np.argsort(norm)[-5:][::-1]
        worst=[]
        unit_scale=1000.*snapshot['cv_j_kg_k'] if f==0 else 1e6 if f in (1,4) else 1.
        for i in candidates:
            worst.append(dict(index=int(i),name=rows[i]['name'],capacity=float(cap[i]),
                origin=float(origin[i,f]),iterate=float(x[i,f]),normalized_residual=float(norm[i]),
                recorded_normalized_residual=float(abs(recorded[i,f])/(cap[i]*tolerance)),
                extensive_residual_j_or_kg=float(residual[i]*unit_scale),
                arithmetic_roundoff_bound_j_or_kg=float(roundoff[i]*unit_scale),
                residual_to_roundoff_bound=float(abs(residual[i])/max(roundoff[i],1e-300))))
        matrix=sparse.diags(cap,format='csr')+sparse.coo_matrix(
            (np.r_[w,w,-w,-w],(np.r_[a,b,a,b],np.r_[a,b,b,a])),shape=(n,n)).tocsr()
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always',MatrixRankWarning)
            solved=spsolve(matrix,cap*origin[:,f])
        forward=None
        if np.isfinite(solved).all():
            factor=1000. if f==0 else 1.
            forward=dict(max_iterate_difference=float(np.max(np.abs(solved-x[:,f])))*factor,
                         capacity_weighted_difference=float(np.sum(cap*np.abs(solved-x[:,f]))/np.sum(cap))*factor)
        result['fields'][name]=dict(max_normalized_residual=float(np.max(norm)),
            max_recorded_normalized_residual=float(np.max(np.abs(recorded[:,f])/(cap*tolerance))),
            max_operator_to_capacity_ratio=float(np.max(matrix.diagonal()/cap)),worst_rows=worst,
            double_sparse_reference=forward,reference_warnings=[str(w.message) for w in notices])
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();result=analyze(json.loads(args.snapshot.read_text()))
    cal.save(args.output,result)
    for name,row in result['fields'].items():
        print(name,'residual',row['max_normalized_residual'],'diag/cap',row['max_operator_to_capacity_ratio'])
        print(json.dumps(row['worst_rows'][0]),flush=True)


if __name__=='__main__':main()
