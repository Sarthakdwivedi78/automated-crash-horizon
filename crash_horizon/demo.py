import numpy as np
import pandas as pd

def demo_data(days=35, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        n = 1600
        t = np.arange(n)/100
        drift = day >= 25
        speed = np.repeat(np.maximum(0,rng.normal(12 if not drift else 18,.5 if not drift else 3,16)),100)
        accel = rng.normal(0,.3 if not drift else 1.2,n)
        if drift:
            accel[900:940] -= 5
        dist = np.full(n,4.)
        if drift:
            dist[900:940] = 1.2
        rows.append(pd.DataFrame({'event_id':f'synthetic-{day}','timestamp':pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=day)+pd.to_timedelta(t,unit='s'),'accel_x_mps2':accel,'speed_mps':speed,'in_work_zone':True,'barrier_distance_m':dist,'label':0 if day == 32 else 2,'source':'SYNTHETIC'}))
    return pd.concat(rows,ignore_index=True)
