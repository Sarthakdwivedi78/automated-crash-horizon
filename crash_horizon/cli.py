import argparse
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from .core import features,daily_metrics,fit_limits,flag_drift,hard_braking
from .demo import demo_data

def main():
    p = argparse.ArgumentParser(description='Offline work-zone monitoring pipeline')
    p.add_argument('--input',help='Normalized telemetry CSV; omit for synthetic demo')
    p.add_argument('--output',default='data/results')
    p.add_argument('--baseline-days',type=int,default=20)
    a = p.parse_args()
    d = features(pd.read_csv(a.input) if a.input else demo_data())
    daily = daily_metrics(d)
    if not 20 <= a.baseline_days < len(daily):
        p.error('Need >=20 baseline days and at least one monitoring day')
    limits = fit_limits(daily.risk.iloc[:a.baseline_days])
    daily['alarm'] = False
    daily.loc[a.baseline_days:,'alarm'] = flag_drift(daily.risk.iloc[a.baseline_days:],limits)
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    daily.to_csv(out/'daily_metrics.csv',index=False)
    hard_braking(d).to_csv(out/'hard_braking.csv',index=False)
    if 'label' in d:
        d[d.label.eq(0) & d.in_work_zone].to_csv(out/'crash_sequences.csv',index=False)
    (out/'run_metadata.json').write_text(json.dumps({'synthetic':not bool(a.input),'limits':asdict(limits),'baseline_days':a.baseline_days,'rows':len(d)},indent=2))
    print(f'Exported results to {out.resolve()}')

if __name__ == '__main__':
    main()
