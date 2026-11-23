# Crash Horizon

A deployment-ready Streamlit research app for work-zone telemetry, daily statistical process control and independent safety-constraint simulation.

**Status:** runnable synthetic demonstration and normalized-data pipeline. No claim of validated crash prediction, work-zone causality or real vehicle safety. No hardware actuator interface is included.

## Run

Use Python 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app starts with 35 reproducible synthetic days, 100 Hz IMU and 1 Hz speed. Drift begins on day 26. No token or dataset download is needed for the demonstration.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m crash_horizon.cli --output data/results
python -m crash_horizon.cli --input data/telemetry.csv --output data/results
```

The CLI exports daily metrics, hard-braking sequences, labeled crash sequences (when supplied) and a run manifest. All runtime data is gitignored.

## Deploy on Streamlit Community Cloud

1. Select this GitHub repository in [Streamlit Community Cloud](https://share.streamlit.io/).
2. Set branch `main`, entrypoint `app.py`, and Python 3.12 under advanced settings.
3. Deploy. The default demo requires no secrets. Never commit tokens or uploaded telemetry.

Dependencies are in `requirements.txt`; `.streamlit/config.toml` supplies the theme and upload limit. CI executes pipeline and Streamlit interaction tests. For a container:

```bash
docker build -t crash-horizon .
docker run --rm -p 8501:8501 crash-horizon
```

See [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). A repository upload is not itself a hosted app deployment.

## Dataset reality and access

[VZCrash](https://huggingface.co/datasets/vzc-research-chapter/VZCrash) is gated and licensed CC BY-NC 4.0. Accept its conditions with your own Hugging Face account. Set `HF_TOKEN` privately or log in using the Hugging Face CLI, then run:

```bash
python scripts/download_data.py --split validation
```

The published schema provides `event_id`, `label`, `vehiclesize`, `gyro_is_hd`, `gsensor`, `gyro`, and `gps_speed`. IMU acceleration is in g at 100 Hz, speed in km/h at 1 Hz. Labels: **0 crash, 1 near miss, 2 normal**. Each event spans 16 seconds. The adapter expects acceleration arrays with shape `(3, samples)` and rejects unrecognized layouts.

The published schema does **not** contain calendar timestamps, GPS coordinates, work-zone geofences, barrier ranges or exact impact times. Accordingly:

- External timestamped GPS/work-zone metadata and barrier-ranging measurements are required for the proposed study.
- The application does not fabricate these fields for real events. `adapt_event` requires an external timezone-aware event start and a sample-aligned geofence mask.
- Event-selected clips cannot estimate unbiased fleet daily crash incidence, exposure rates or causal effects of construction zones.
- Dataset access and actual-data integration have not been validated against gated files. The adapter is tested against the documented schema using fixtures.

To normalize a local parquet after obtaining synchronized external context:

```bash
python scripts/normalize_vzcrash.py data/events.parquet data/context.csv data/telemetry.csv
```

Context CSV columns: `event_id,sample_index,timestamp,in_work_zone` and optionally `barrier_distance_m`. Provide every sample for each selected event, with zero-based contiguous indices. The converter checks timestamps against the 100 Hz timeline and skips events with no external context. Process one shard at a time to bound memory.

Do not redistribute downloaded data in this repository. Source code licensing does not replace the dataset license. Cite Bianconcini et al., *VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes*, ITSC 2026.

## Normalized input contract

| Column | Meaning |
|---|---|
| `event_id` | Unique clip/session identifier; never reused for another vehicle |
| `timestamp` | ISO-8601 UTC sample timestamp, nominal 10 ms spacing |
| `accel_x_mps2` | Calibrated forward acceleration, m/s²; braking is negative |
| `speed_mps` | Nonnegative speed, m/s; 1 Hz samples held forward, no backward interpolation |
| `in_work_zone` | Verified geofence membership as boolean or 0/1 |
| `barrier_distance_m` | Optional synchronized ranging distance; absence produces sensor fault |
| `label` | Optional VZCrash event label; not an impact timestamp |

Upload at most 500,000 rows / 50 MB. Large datasets should be processed offline. The daily view requires at least 20 eligible baseline days plus one monitoring day. Download a schema-compatible synthetic CSV from the app. `geofence.inside_polygon` provides boundary-inclusive membership for small local WGS84 polygons; it is not a ranging sensor and does not support holes or antimeridian-crossing polygons. Upstream joins must account for GPS age, uncertainty and work-zone activation dates.

## Methodology

1. Validate, sort and split telemetry by event, gaps and geofence transitions. Use causal rolling windows; never cross a clip boundary.
2. Braking proxy `max(-accel_x, 0)`: 100-sample rolling sample standard deviation (`ddof=1`). This is acceleration, not measured force; mass is unavailable.
3. Speed SD: three trailing 1 Hz observations, also `ddof=1`. Repeated 100 Hz held values are not independent GPS measurements.
4. Risk index: `0.5 × brake_SD / 3 + 0.3 × speed_SD / 3 + 0.2 × speed / 13.89`. References have matching units. Weights and references are illustrative, not learned or calibrated. Daily scores average eligible in-zone samples; incomplete and event-selected exposure must be reported.
5. Fit frozen limits to the first baseline days only. Default sigma is `sample_SD / c4(n)`, where sample variance uses n−1. Alternative classic I-MR sigma is `mean(abs(diff(x))) / 1.128`. Center ±3 sigma; do not divide by sqrt(n). Lower limit is not clipped. See [NIST Individuals charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm).
6. Flag monitoring points outside limits or eight successive points on one side. Never refit on alarm periods automatically. Baseline stability, autocorrelation and false-alarm calibration require review.
7. Isolation Forest trains only on baseline in-zone features. It is an anomaly detector, **not** a calibrated collision model. No accuracy/AUC claims are reported without an appropriate labeled evaluation.
8. Extract in-zone braking below −3 m/s² lasting at least 100 ms. Threshold is a research parameter, not a universal vehicle standard.

Daily SPC is retrospective. Telemetry replay demonstrates per-sample decisions but is not a live 100 Hz ingestion service. A fixed 1.5 m range cannot guarantee stopping at arbitrary speeds. The software does not enforce safe vehicle speed.

## STAMP-inspired safety analysis

See [SAFETY.md](SAFETY.md) for controller structure, unsafe control actions and validation requirements. The executable rule is a constraint demonstrator, not a complete STAMP/STPA analysis or certified emergency braking controller.

## Research validation plan

Collect representative continuous in-zone and matched out-of-zone exposure, stable construction activity intervals, vehicle identifiers, synchronized ranging, and impact timestamps. Account for vehicle, road, weather, traffic and sampling artifacts before estimating zone effects; variance association alone does not prove causality or artificial telemetry drift. Freeze training and thresholds; split by vehicle and time; preserve source train/validation/test separation. Evaluate pre-impact horizons using only preceding data, event PR-AUC, recall at a fixed false-alarm rate per driving hour, warning lead-time distributions and confidence intervals. Without impact times, only event detection is supportable.

## Layout

- `app.py`: dashboard, CSV upload, replay, constraint simulator and downloads
- `crash_horizon/core.py`: validation, features, SPC, anomaly detection, braking extraction
- `crash_horizon/vzcrash.py`: documented-schema adapter with external context
- `crash_horizon/geofence.py`: local polygon membership
- `crash_horizon/cli.py`: offline pipeline and exports
- `tests/`: numerical, boundary, leakage and Streamlit interaction checks
- `.github/workflows/tests.yml`: automated CI
