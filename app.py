import json
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from crash_horizon.core import features, daily_metrics, fit_limits, flag_drift, fit_anomaly_model, MODEL_COLUMNS, hard_braking, safety_check
from crash_horizon.demo import demo_data

st.set_page_config(page_title='Crash Horizon | Work-zone observatory',page_icon='🛣️',layout='wide')
st.title('Crash Horizon')
st.caption('WORK-ZONE OBSERVATORY · Process drift, telemetry and independent safety constraints')
st.warning('Research simulator — no hardware control, calibrated crash probability or validated collision forecast.')
with st.sidebar:
    st.header('Monitoring setup')
    source = st.radio('Data source',['Synthetic demonstration','Upload normalized telemetry'])
    uploaded = st.file_uploader('Telemetry CSV',type=['csv']) if source.startswith('Upload') else None
    baseline_days = st.slider('Baseline days (frozen limits)',20,25,20)
    method = st.selectbox('I-chart estimator',['sample','moving_range'])
    st.caption('Sample: n−1 variance with c₄ correction. Moving range: MR̄ / 1.128.')
    st.markdown('[Dataset and access conditions](https://huggingface.co/datasets/vzc-research-chapter/VZCrash)')
@st.cache_data
def get_demo():
    return demo_data()
if source.startswith('Upload') and uploaded is None:
    st.info('Upload normalized CSV. Required: event_id, timestamp (UTC), accel_x_mps2, speed_mps, in_work_zone. Optional: barrier_distance_m, label.')
    st.stop()
try:
    raw = get_demo() if source.startswith('Synthetic') else pd.read_csv(uploaded)
    if len(raw) > 500_000:
        raise ValueError('Interactive upload limit: 500,000 rows. Aggregate larger datasets offline.')
    d = features(raw)
    daily = daily_metrics(d)
    if len(daily) <= baseline_days:
        raise ValueError('Need at least one monitoring day after the baseline (21+ eligible days). VZCrash dates require external context.')
    limits = fit_limits(daily.risk.iloc[:baseline_days],method)
except (ValueError,KeyError,TypeError) as exc:
    st.error(str(exc))
    st.stop()
cutoff = daily.day.iloc[baseline_days]
daily['phase'] = np.where(daily.day < cutoff,'Baseline','Monitoring')
daily['alarm'] = False
daily.loc[daily.phase.eq('Monitoring'),'alarm'] = flag_drift(daily.loc[daily.phase.eq('Monitoring'),'risk'],limits)
model = fit_anomaly_model(d[(d.timestamp < cutoff) & d.in_work_zone])
d['ml_hazard'] = pd.Series(pd.NA,index=d.index,dtype='boolean')
valid = d[MODEL_COLUMNS].notna().all(axis=1) & d.in_work_zone
d.loc[valid,'ml_hazard'] = model.predict(d.loc[valid,MODEL_COLUMNS]) == -1
brakes = hard_braking(d)
cols = st.columns(4)
cols[0].metric('Telemetry samples',f'{len(d):,}')
cols[1].metric('Monitoring alarms',int(daily.alarm.sum()))
cols[2].metric('Hard-braking sequences',len(brakes))
cols[3].metric('Baseline days',baseline_days)
st.caption('SYNTHETIC DATA · Programmed drift starts on day 26. Results are illustrative.' if source.startswith('Synthetic') else 'USER DATA · Verify units, axis alignment, timestamps, sampling and context provenance.')
tabs = st.tabs(['Process control','Telemetry replay','Constraint simulator','Data & methods'])
with tabs[0]:
    fig = go.Figure(go.Scatter(x=daily.day,y=daily.risk,mode='lines+markers',name='Daily risk index',line_color='#19C6B5'))
    for y,name,color in [(limits.ucl,'UCL','#FF6B7A'),(limits.center,'Center','#91A5C6'),(limits.lcl,'LCL','#FF6B7A')]:
        fig.add_hline(y=y,line_dash='dash',line_color=color,annotation_text=name)
    alarm = daily[daily.alarm]
    fig.add_trace(go.Scatter(x=alarm.day,y=alarm.risk,mode='markers',name='Drift alarm',marker=dict(color='#FF6B7A',size=13)))
    fig.update_layout(height=410,yaxis_title='Dimensionless risk index',xaxis_title='UTC day',legend_orientation='h')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Frozen baseline limits. Alarms: outside 3σ or eight consecutive monitoring points on one side. Daily aggregates are retrospective, not a real-time pre-crash response.')
    st.dataframe(daily,hide_index=True,use_container_width=True)
    st.download_button('Export daily metrics',daily.to_csv(index=False),'daily_metrics.csv','text/csv')
with tabs[1]:
    event = st.selectbox('Event',d.event_id.unique())
    g = d[d.event_id.eq(event)].reset_index(drop=True)
    pos = st.slider('Replay through sample',1,len(g),len(g))
    view = g.iloc[:pos]
    st.line_chart(view.set_index('timestamp')[['speed_mps','brake_mps2']])
    st.caption('Recorded replay, not a live sensor feed. Speed is held forward at its native 1 Hz rate.')
    st.dataframe(brakes[brakes.event_id.eq(event)],hide_index=True)
    last = view.iloc[-1]
    hazard = None if pd.isna(last.ml_hazard) else bool(last.ml_hazard)
    st.code(safety_check(last.get('barrier_distance_m',None),hazard,bool(last.in_work_zone)))
    st.caption('Isolation Forest detects baseline anomalies; it does not estimate crash probability. Missing barrier context produces a sensor-fault state.')
    if 'label' in g:
        st.write('Event label:',{0:'Crash',1:'Near miss',2:'Normal driving'}.get(int(g.label.iloc[0]),'Unknown'))
with tabs[2]:
    st.subheader('Independent STAMP-inspired constraint')
    distance = st.number_input('Barrier distance (m)',min_value=0.,value=1.2,step=.1)
    hazard = st.checkbox('ML flagged a hazard',value=False)
    zone = st.checkbox('Inside verified work-zone geofence',value=True)
    age = st.number_input('Distance sensor age (seconds)',min_value=0.,value=0.,step=.1)
    state = safety_check(distance,hazard,zone,age)
    st.code(state)
    st.info('Distance < 1.5 m AND no ML hazard → simulated emergency override. Exactly 1.5 m does not trigger. Missing or stale measurements → sensor fault requiring a separately validated fallback.')
    st.download_button('Export decision audit',json.dumps(dict(distance_m=distance,ml_hazard=hazard,in_work_zone=zone,sensor_age_s=age,decision=state,simulation=True),indent=2),'constraint_audit.json')
with tabs[3]:
    st.markdown('''**Risk index:** 0.5 × rolling brake SD / 3 m/s² + 0.3 × trailing speed SD / 3 m/s + 0.2 × speed / 13.89 m/s. Weights and references are uncalibrated research choices. Braking acceleration is a force proxy; vehicle mass is unavailable.

**No leakage:** trailing windows, event/gap/geofence resets, frozen baseline limits, baseline-only anomaly training. Label 0 means crash, 1 near miss, 2 normal. Event labels do not specify impact time.

**Dataset gap:** published VZCrash has no calendar timestamps, coordinates, construction polygons or barrier ranging. External synchronized context is required. It cannot alone establish work-zone causality or warning lead time.

**Validation needed:** continuous exposure, stable baseline, autocorrelation analysis, vehicle-disjoint testing, impact timestamps, and false alarms per driving hour. Event-selected data cannot estimate fleet daily crash incidence.''')
    st.json(asdict(limits))
    st.download_button('Download normalized demo CSV',get_demo().to_csv(index=False),'synthetic_telemetry.csv','text/csv')
