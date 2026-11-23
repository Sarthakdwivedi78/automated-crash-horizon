# STAMP-inspired research constraint

Losses: collision injury, secondary collision caused by inappropriate braking, and loss of control.

Control structure: ranging/GPS/IMU sensors → estimator and geofence monitor → ML anomaly detector → independent constraint monitor → **simulated brake request** → recorded state. There is no actuator transport. A deployed controller would require authenticated commands, actuator acknowledgement, watchdogs and feedback about actual deceleration.

| Unsafe control action | Constraint / prototype behavior |
|---|---|
| No response below 1.5 m when ML misses hazard | In zone, valid range <1.5 m and ML false/unavailable returns `SIMULATED_EMERGENCY_OVERRIDE` |
| ML detects close hazard but action omitted | Return `MODEL_HAZARD_BRAKE_REQUEST`; prototype cannot confirm actuation |
| Stale, invalid or missing range interpreted as safe | Return `SENSOR_FAULT_REVIEW`; never return normal monitoring |
| Spurious emergency braking | Validate ranging confidence, obstacle trajectory, friction and following traffic before real implementation |
| Command arrives too late or persists too long | Requires vehicle-specific latency budget, stopping-distance model, latch/release and acknowledgement design |

The range comparison is strictly `<1.5`; exactly 1.5 does not trigger. Sensor age above 0.2 seconds is considered stale for this demonstrator; this is an unvalidated configuration assumption. Unknown ML output counts as a missed flag when valid range is close. Zone membership and timestamps are assumed verified upstream. Range to a zone polygon edge is not distance to a physical barrier.

This prototype does not provide functional safety certification, fail-operational behavior, obstacle classification, braking torque control, safe-speed computation or hardware integration. Production emergency braking needs a full hazard analysis, independent safety review, hardware-in-the-loop and closed-track validation, and vehicle-specific regulatory and engineering work. The 1.5 m threshold alone is insufficient to guarantee stopping distance.
