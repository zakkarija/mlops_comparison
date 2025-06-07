from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String, Int64

# Define equipment entity
equipment = Entity(
    name="equipment",
    join_keys=["equipment_id"],
)

repo_root = Path(__file__).parent            # …/feature_repo
f3_timeseries_source = FileSource(
    path=str(repo_root / "data/offline/f3_timeseries.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Time series features with rolling windows
f3_timeseries_features = FeatureView(
    name="f3_timeseries_features",
    entities=[equipment],
    ttl=timedelta(minutes=30),
    schema=[
        Field(name="f3_current", dtype=Float32),
        Field(name="f3_rolling_mean_10", dtype=Float32),
        Field(name="f3_rolling_mean_50", dtype=Float32),
        Field(name="f3_rolling_mean_100", dtype=Float32),
        Field(name="f3_rolling_std_10", dtype=Float32),
        Field(name="f3_rolling_std_50", dtype=Float32),
        Field(name="f3_rolling_min_50", dtype=Float32),
        Field(name="f3_rolling_max_50", dtype=Float32),
        Field(name="f3_rate_of_change", dtype=Float32),
        Field(name="movement_direction", dtype=Int64),
        Field(name="cycle_position", dtype=Float32),
        Field(name="anomaly_class", dtype=Int64),
        Field(name="label", dtype=String),
    ],
    source=f3_timeseries_source,
)