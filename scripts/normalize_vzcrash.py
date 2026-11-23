"""Join a local VZCrash parquet to explicit external sample context.

Run from repository root: python scripts/normalize_vzcrash.py events.parquet context.csv data/telemetry.csv
Context columns: event_id,sample_index,timestamp,in_work_zone[,barrier_distance_m].
Every event's samples must be present, ordered by index and aligned at 100 Hz.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from crash_horizon.vzcrash import adapt_event
from crash_horizon.core import validate

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parquet')
    p.add_argument('context')
    p.add_argument('output')
    args = p.parse_args()
    events = pd.read_parquet(args.parquet)
    context = pd.read_csv(args.context,dtype={'event_id':str})
    required = {'event_id','sample_index','timestamp','in_work_zone'}
    if not required.issubset(context):
        p.error(f'Context requires {sorted(required)}')
    rows = []
    for _, event in events.iterrows():
        c = context[context.event_id.eq(str(event.event_id))].sort_values('sample_index')
        if c.empty:
            continue
        if not np.array_equal(c.sample_index,np.arange(len(c))):
            p.error('Context sample indices must be complete and unique, starting at zero')
        d = adapt_event(event,c.timestamp.iloc[0],c.in_work_zone.to_numpy(),c.barrier_distance_m.to_numpy() if 'barrier_distance_m' in c else None)
        if not np.array_equal(pd.to_datetime(c.timestamp,utc=True).to_numpy(),d.timestamp.to_numpy()):
            p.error('External context timestamps do not match the 100 Hz event timeline')
        rows.append(d)
    if not rows:
        p.error('No events matched external context')
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    validate(pd.concat(rows,ignore_index=True)).to_csv(out,index=False)
    print(f'Normalized {len(rows)} matched events to {out}')

if __name__ == '__main__':
    main()
