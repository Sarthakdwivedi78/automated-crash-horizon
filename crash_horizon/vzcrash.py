"""Adapter for published nested event schema; context is never invented."""
import numpy as np
import pandas as pd

def adapt_event(event, start_timestamp, zone_mask, barrier_distance_m=None):
    a = np.asarray(event['gsensor'],dtype=float)
    if a.ndim != 2 or a.shape[0] != 3:
        raise ValueError('Expected gsensor shape (3, samples); inspect source schema')
    n = a.shape[1]
    speed = np.asarray(event['gps_speed'],dtype=float)
    if speed.ndim != 1 or len(speed) != int(np.ceil(n/100)):
        raise ValueError('GPS/IMU length mismatch')
    zone = np.asarray(zone_mask)
    if len(zone) != n:
        raise ValueError('External geofence mask must match IMU length')
    start = pd.Timestamp(start_timestamp)
    if start.tzinfo is None:
        raise ValueError('External event start requires timezone')
    d = pd.DataFrame({'event_id':str(event['event_id']),'timestamp':start+pd.to_timedelta(np.arange(n)/100,unit='s'),'accel_x_mps2':a[0]*9.80665,'speed_mps':np.repeat(speed,100)[:n]/3.6,'in_work_zone':zone,'label':int(event['label']),'source':'VZCrash + external context'})
    if barrier_distance_m is not None:
        d['barrier_distance_m'] = barrier_distance_m
    return d
