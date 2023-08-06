import os

from utils import Team9


config = {
    "id_name": "icu_id",
    "label": None,
    "feat_timevarying": ["oxygenation___custom"],
    "feat_static": ["sex", "race", "bmi_value", "WHO_Region"],
    "metric": "custom_dtw",  # custom dynamic time warping HDBSCAN clusterer
    "K_time": 5,
    "scaler": None,
    "cap_datasets": 200,
    "fillna_strategy": "fill_forward",
    "tte_name": None,
    "time_name": None,
    "seed": 42,
    "test_size": (1 - 5 / 100),
    "is_bigquery": False,
    "query_or_path": os.path.join("data", "coredata.csv"),
}


if __name__ == "__main__":
    project = Team9(**config)
    df = project.df
    print(df.head(5))
    print(f"Number of patients: {len(df[project.id_name].unique())}")  # 92244
    # Train clustering
    project.run_clustering()
    # Run some analysis
    project.analyze_clusters()
