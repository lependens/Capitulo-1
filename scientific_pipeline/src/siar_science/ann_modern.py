"""ANN v1 training primitives. No dataset or TEST access in this module."""
from __future__ import annotations

import hashlib
import json
import math
from statistics import median

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def validate_ann_contract(ann, ml, *, ann_bytes=None, ml_bytes=None):
    if ann.get('schema_version') != 'ann-modern-et0-v1':
        raise ValueError('ANN contract version mismatch')
    if ann.get('status') != 'FROZEN':
        raise ValueError('ANN contract not frozen')
    if ml_bytes is not None and hashlib.sha256(ml_bytes).hexdigest() != ann['ml_protocol_sha256']:
        raise ValueError('ML protocol hash mismatch')
    if ann['dataset_sha256'] != ml['dataset_sha256'] or ann['target'] != ml['target']:
        raise ValueError('Dataset or target mismatch')
    if ann['feature_sets'] != list(ml['feature_sets']):
        raise ValueError('Feature sets changed')
    for a, b in zip(ann['folds'], ml['development_folds'], strict=True):
        if a['id'] != b['fold_id'] or a['train'] != b['train'] or a['validation'] != b['validation']:
            raise ValueError('Temporal fold mismatch')
    if ann['development'] != ml['development_period'] or ann['test'] != ml['test_period']:
        raise ValueError('Temporal period mismatch')
    if len(ann['seeds']) != len(set(ann['seeds'])) or not ann['seeds']:
        raise ValueError('Seeds must be distinct and fixed')
    if ann['test_status'] != 'retrospective_locked':
        raise ValueError('TEST status mismatch')
    return hashlib.sha256(ann_bytes).hexdigest() if ann_bytes is not None else None


def _matrix(rows, family):
    eligible = [r for r in rows if r['features'][family] is not None]
    x = np.asarray([r['features'][family] for r in eligible], dtype=np.float64)
    y = np.asarray([r['target'] for r in eligible], dtype=np.float64)
    if not len(eligible) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Empty or nonfinite training matrix')
    return eligible, x, y


def _model(widths, seed, ann):
    return MLPRegressor(hidden_layer_sizes=tuple(widths), activation=ann['activation'],
                        solver='adam', alpha=ann['alpha'], batch_size=ann['batch_size'],
                        learning_rate_init=ann['learning_rate_init'], random_state=seed,
                        shuffle=False, early_stopping=False)


def _epochs(model, x, y, n):
    for _ in range(n):
        model.partial_fit(x, y)
    return model


def _inner_dates(rows, fraction=0.15):
    dates = sorted({r['date'] for r in rows})
    cutoff = min(len(dates)-1, max(1, math.floor(len(dates)*(1-fraction))))
    return dates[cutoff]


def fit_seed_fold(train_rows, validation_rows, family, widths, seed, ann):
    """Choose epochs inside TRAIN, then refit on whole fold TRAIN."""
    train, _, _ = _matrix(train_rows, family)
    val, xv, yv = _matrix(validation_rows, family)
    cutoff = _inner_dates(train)
    inner_fit = [r for r in train if r['date'] < cutoff]
    inner_val = [r for r in train if r['date'] >= cutoff]
    _, xi, yi = _matrix(inner_fit, family)
    _, xh, yh = _matrix(inner_val, family)
    scaler = StandardScaler().fit(xi)
    model = _model(widths, seed, ann)
    best_rmse = float('inf')
    best_epoch = 0
    stale = 0
    cfg = ann['early_stopping']
    for epoch in range(1, cfg['max_epochs']+1):
        model.partial_fit(scaler.transform(xi), yi)
        pred = model.predict(scaler.transform(xh))
        rmse = float(np.sqrt(np.mean((pred-yh)**2)))
        if rmse < best_rmse-cfg['min_delta_rmse']:
            best_rmse, best_epoch, stale = rmse, epoch, 0
        else:
            stale += 1
        if stale >= cfg['patience_epochs']:
            break
    _, xf, yf = _matrix(train, family)
    full_scaler = StandardScaler().fit(xf)
    full_model = _epochs(_model(widths, seed, ann), full_scaler.transform(xf), yf, best_epoch)
    pv = full_model.predict(full_scaler.transform(xv))
    return {'seed': seed, 'best_epoch': best_epoch, 'inner_rmse': best_rmse,
            'validation_keys': [(r['station_code'], r['date']) for r in val],
            'validation_target': yv, 'validation_prediction': pv}


def fold_candidate(rows, fold, family, widths, ann):
    train = [r for r in rows if r['split']=='DEVELOPMENT' and fold['train'][0] <= r['date'] <= fold['train'][1] and r['features'][family] is not None]
    validation = [r for r in rows if r['split']=='DEVELOPMENT' and fold['validation'][0] <= r['date'] <= fold['validation'][1] and all(r['features'][fam] is not None for fam in ann['feature_sets'])]
    if not train or not validation:
        raise ValueError('Empty ANN fold')
    fits = [fit_seed_fold(train, validation, family, widths, seed, ann) for seed in ann['seeds']]
    keys = fits[0]['validation_keys']
    if any(f['validation_keys'] != keys for f in fits):
        raise ValueError('Seed cohort mismatch')
    y = fits[0]['validation_target']
    pred = np.mean([f['validation_prediction'] for f in fits], axis=0)
    return {'fold_id':fold['id'], 'n':len(y),
            'rmse':float(np.sqrt(np.mean((pred-y)**2))),
            'mae':float(np.mean(np.abs(pred-y))),
            'bias':float(np.mean(pred-y)),
            'seed_rmse':[float(np.sqrt(np.mean((f['validation_prediction']-y)**2))) for f in fits],
            'best_epochs':[f['best_epoch'] for f in fits]}


def summarize_folds(folds):
    n = np.asarray([x['n'] for x in folds], dtype=float)
    r = np.asarray([x['rmse'] for x in folds], dtype=float)
    mean = float(np.average(r, weights=n))
    return {'n':int(n.sum()), 'weighted_mean_rmse':mean,
            'weighted_sd_rmse':float(np.sqrt(np.average((r-mean)**2, weights=n))),
            'pooled_rmse':float(np.sqrt(np.average(r*r, weights=n))),
            'min_fold_rmse':float(r.min()), 'max_fold_rmse':float(r.max())}


def final_epochs(folds, seeds):
    return {str(seed):math.floor(median(f['best_epochs'][i] for f in folds)+0.5)
            for i,seed in enumerate(seeds)}
