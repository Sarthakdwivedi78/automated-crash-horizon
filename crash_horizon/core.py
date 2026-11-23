"""Causal telemetry features and frozen Phase-II process monitoring."""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED = ['event_id', 'timestamp', 'accel_x_mps2', 'speed_mps', 'in_work_zone']

def validate(df):
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    d = df.copy()
    if d[REQUIRED].isna().any().any():
        raise ValueError('Required telemetry cannot contain nulls')
    d['timestamp'] = pd.to_datetime(d.timestamp, utc=True, errors='raise')
    if d.timestamp.isna().any():
        raise ValueError('Invalid timestamps')
    for c in ['accel_x_mps2', 'speed_mps']:
        d[c] = pd.to_numeric(d[c], errors='raise')
        if not np.isfinite(d[c]).all():
            raise ValueError(f'Nonfinite {c}')
    if (d.speed_mps < 0).any():
        raise ValueError('Speed must be nonnegative')
    if not d.in_work_zone.isin([True, False, 0, 1]).all():
        raise ValueError('in_work_zone must be boolean or 0/1')
    d['in_work_zone'] = d.in_work_zone.astype(bool)
    if d.duplicated(['event_id', 'timestamp']).any():
        raise ValueError('Duplicate event timestamps')
    d = d.sort_values(['event_id', 'timestamp']).reset_index(drop=True)
    delta = d.groupby('event_id').timestamp.diff().dt.total_seconds().dropna()
    if ((delta < .0099) | (np.abs(delta/.01 - np.round(delta/.01)) > .01)).any():
        raise ValueError('Expected nominal 100 Hz timestamps; resample and validate upstream')
    if 'label' in d and not d.label.isin([0,1,2]).all():
        raise ValueError('Labels must be 0 (crash), 1 (near miss), or 2 (normal)')
    return d

def features(df, window=100, speed_reference=13.89):
    if window < 2 or speed_reference <= 0:
        raise ValueError('Invalid feature configuration')
    d = validate(df)
    pieces = []
    for _, g in d.groupby('event_id', sort=False):
        g = g.copy()
        gap = g.timestamp.diff().dt.total_seconds()
        segment = ((gap > .015) | g.in_work_zone.ne(g.in_work_zone.shift())).cumsum()
        g['brake_mps2'] = (-g.accel_x_mps2).clip(lower=0)
        g['brake_sd'] = g.groupby(segment).brake_mps2.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=1))
        # Compute speed variance at native 1 Hz, without treating held samples as independent.
        sec = ((g.timestamp-g.timestamp.iloc[0]).dt.total_seconds()+1e-8).astype(int)
        speeds = g.groupby([segment, sec], sort=True).speed_mps.first()
        sd = speeds.groupby(level=0).transform(lambda s: s.rolling(3, min_periods=3).std(ddof=1))
        keys = pd.MultiIndex.from_arrays([segment, sec])
        g['speed_sd'] = sd.reindex(keys).to_numpy()
        g['risk'] = .5*g.brake_sd/3 + .3*g.speed_sd/3 + .2*g.speed_mps/speed_reference
        pieces.append(g)
    if not pieces:
        raise ValueError('No telemetry rows')
    return pd.concat(pieces, ignore_index=True)

def daily_metrics(d):
    z = d[d.in_work_zone & d.risk.notna()].copy()
    z['day'] = z.timestamp.dt.floor('D')
    return z.groupby('day').agg(risk=('risk','mean'), samples=('risk','size'), events=('event_id','nunique')).reset_index()

@dataclass(frozen=True)
class ControlLimits:
    center: float
    sigma: float
    lcl: float
    ucl: float
    method: str

def fit_limits(values, method='sample'):
    x = np.asarray(values, dtype=float)
    if len(x) < 20 or not np.isfinite(x).all():
        raise ValueError('At least 20 finite baseline daily observations required')
    if method == 'sample':
        n = len(x)
        c4 = math.sqrt(2/(n-1))*math.exp(math.lgamma(n/2)-math.lgamma((n-1)/2))
        sigma = x.std(ddof=1)/c4
    elif method == 'moving_range':
        sigma = np.abs(np.diff(x)).mean()/1.128
    else:
        raise ValueError('Unknown limit estimator')
    if sigma <= 1e-12:
        raise ValueError('Baseline variation is zero; collect a representative baseline')
    center = float(x.mean())
    return ControlLimits(center,float(sigma),center-3*sigma,center+3*sigma,method)

def flag_drift(values, limits):
    x = pd.Series(values).reset_index(drop=True)
    upper, lower = x > limits.center, x < limits.center
    run = upper.rolling(8).sum().eq(8) | lower.rolling(8).sum().eq(8)
    return ((x > limits.ucl) | (x < limits.lcl) | run).to_numpy()

MODEL_COLUMNS = ['brake_sd','speed_sd','speed_mps']
def fit_anomaly_model(baseline):
    x = baseline[MODEL_COLUMNS].dropna()
    if len(x) < 100:
        raise ValueError('Not enough baseline features')
    return IsolationForest(n_estimators=100,contamination=.02,random_state=42,n_jobs=1).fit(x)

def hard_braking(d, threshold=-3.0, minimum_seconds=.1):
    rows = []
    for event, g in d.groupby('event_id'):
        mask = (g.accel_x_mps2 <= threshold) & g.in_work_zone
        gap = g.timestamp.diff().dt.total_seconds().gt(.015)
        groups = (mask.ne(mask.shift()) | gap).cumsum()
        for _, b in g[mask].groupby(groups[mask]):
            duration = (b.timestamp.iloc[-1]-b.timestamp.iloc[0]).total_seconds()+.01
            if duration >= minimum_seconds:
                rows.append(dict(event_id=event,start=b.timestamp.iloc[0],end=b.timestamp.iloc[-1],duration_s=duration,peak_deceleration_mps2=-b.accel_x_mps2.min()))
    return pd.DataFrame(rows,columns=['event_id','start','end','duration_s','peak_deceleration_mps2'])

def safety_check(distance_m, ml_hazard, in_work_zone=True, sensor_age_s=0.0):
    """Pure decision function; no actuator connection."""
    if not in_work_zone:
        return 'OUTSIDE_ZONE'
    if distance_m is None or not np.isfinite(distance_m) or distance_m < 0 or not np.isfinite(sensor_age_s) or sensor_age_s < 0 or sensor_age_s > .2:
        return 'SENSOR_FAULT_REVIEW'
    flagged = isinstance(ml_hazard, (bool, np.bool_)) and bool(ml_hazard)
    if distance_m < 1.5 and not flagged:
        return 'SIMULATED_EMERGENCY_OVERRIDE'
    if distance_m < 1.5:
        return 'MODEL_HAZARD_BRAKE_REQUEST'
    return 'MONITOR'
