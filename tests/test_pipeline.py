import numpy as np
import pandas as pd
import pytest
from crash_horizon.core import features,daily_metrics,fit_limits,flag_drift,safety_check,hard_braking
from crash_horizon.demo import demo_data
from crash_horizon.vzcrash import adapt_event
from crash_horizon.geofence import inside_polygon

@pytest.mark.parametrize('distance,hazard,expected',[(1.49,False,'SIMULATED_EMERGENCY_OVERRIDE'),(1.5,False,'MONITOR'),(1.49,True,'MODEL_HAZARD_BRAKE_REQUEST'),(None,False,'SENSOR_FAULT_REVIEW'),(float('nan'),False,'SENSOR_FAULT_REVIEW'),(1.0,None,'SIMULATED_EMERGENCY_OVERRIDE')])
def test_constraint(distance,hazard,expected):
    assert safety_check(distance,hazard) == expected

def test_stale_and_outside():
    assert safety_check(1.,False,sensor_age_s=.21) == 'SENSOR_FAULT_REVIEW'
    assert safety_check(1.,False,in_work_zone=False) == 'OUTSIDE_ZONE'
    assert safety_check(1.,np.bool_(True)) == 'MODEL_HAZARD_BRAKE_REQUEST'

def test_gap_resets_and_sampling():
    raw = demo_data(1)
    gapped = raw.drop(index=range(500,520))
    f = features(gapped)
    assert f.loc[500:598,'brake_sd'].isna().all()
    raw.loc[1,'timestamp'] += pd.Timedelta(milliseconds=1)
    with pytest.raises(ValueError):
        features(raw)

def test_causal_features_and_reset():
    raw = demo_data(2)
    a = features(raw)
    changed = raw.copy()
    changed.loc[changed.index >= 1000,'accel_x_mps2'] = -100
    b = features(changed)
    pd.testing.assert_frame_equal(a.iloc[:1000],b.iloc[:1000])
    assert a.groupby('event_id').head(99).brake_sd.isna().all()
    expected = (-raw.accel_x_mps2.iloc[:100]).clip(lower=0).std(ddof=1)
    assert a.brake_sd.iloc[99] == pytest.approx(expected)

def test_sample_limits_and_drift():
    x = np.arange(20,dtype=float)
    limits = fit_limits(x)
    assert limits.center == 9.5
    assert limits.sigma > x.std(ddof=1)
    assert flag_drift([limits.center]*7+[limits.ucl+1],limits)[-1]
    assert flag_drift([limits.center+.1]*8,limits)[-1]
    with pytest.raises(ValueError):
        fit_limits(np.ones(20))

def test_demo_drift_and_braking():
    d = features(demo_data())
    daily = daily_metrics(d)
    limits = fit_limits(daily.risk.iloc[:20])
    assert flag_drift(daily.risk.iloc[25:],limits).all()
    assert len(hard_braking(d)) >= 1

def test_adapter_units_and_labels():
    e = {'event_id':1,'gsensor':[[1]*200,[0]*200,[1]*200],'gps_speed':[36,72],'label':0}
    d = adapt_event(e,'2026-01-01T00:00:00Z',[True]*200)
    assert d.accel_x_mps2.iloc[0] == pytest.approx(9.80665)
    assert d.speed_mps.iloc[99] == 10
    assert d.speed_mps.iloc[100] == 20
    assert d.label.eq(0).all()
    with pytest.raises(ValueError):
        adapt_event(e,'2026-01-01',[True]*200)

def test_polygon():
    ring = [[0,0],[1,0],[1,1],[0,1],[0,0]]
    assert inside_polygon(.5,.5,ring)
    assert inside_polygon(0,.5,ring)
    assert not inside_polygon(2,.5,ring)

def test_bad_input():
    raw = demo_data(1)
    raw.loc[0,'speed_mps'] = -1
    with pytest.raises(ValueError):
        features(raw)
