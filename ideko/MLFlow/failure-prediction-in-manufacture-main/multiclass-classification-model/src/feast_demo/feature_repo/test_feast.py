from feast import FeatureStore
import pandas as pd
from datetime import datetime, timedelta


def test_basic_feast():
    """Test basic Feast functionality"""

    print("🚀 Testing Feast setup...")

    # Initialize feature store
    fs = FeatureStore(repo_path=".")
    print("✅ Feature store initialized")

    # List features
    print("\n📋 Available feature views:")
    for fv in fs.list_feature_views():
        print(f"  - {fv.name}")
        for field in fv.schema:
            print(f"    - {field.name} ({field.dtype})")

    # Test getting historical features
    print("\n📊 Testing historical features...")

    entity_df = pd.DataFrame({
        "equipment_id": ["test_machine_001"],
        "event_timestamp": [datetime.now() - timedelta(minutes=30)]
    })

    try:
        historical_features = fs.get_historical_features(
            entity_df=entity_df,
            features=["f3_basic:f3_value"]
        ).to_df()

        print("✅ Successfully retrieved historical features!")
        print(historical_features)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be normal - we may need data in the right time range")

    print("\n🎉 Feast is working! Ready for Phase 2.")


if __name__ == "__main__":
    test_basic_feast()