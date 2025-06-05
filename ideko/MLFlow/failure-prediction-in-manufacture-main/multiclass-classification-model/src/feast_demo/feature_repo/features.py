from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

# Define equipment entity (each zip file becomes a unique equipment)
equipment = Entity(
    name="equipment",
    join_keys=["equipment_id"],
)

# Define data source pointing to your processed training data
f3_source = FileSource(
    name="f3_training_data_source",
    path="data/f3_basic.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define feature view with f3 value and label
f3_basic_features = FeatureView(
    name="f3_basic",
    entities=[equipment],
    ttl=timedelta(days=30),  # Keep features for 30 days
    schema=[
        Field(name="f3_value", dtype=Float32),
        Field(name="label", dtype=String),  # Add label field
    ],
    source=f3_source,
)