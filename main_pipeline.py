import pandas as pd # type: ignore
import numpy as np
import ray
import mlflow
from prefect import task, flow
from src.features import create_features, add_target_encoding
from src.trainer import train_family_model
import sys

@task
def load_data():
    # Load all necessary files for the merge
    train = pd.read_csv('data/train.csv', parse_dates=['date'])
    test = pd.read_csv('data/test.csv', parse_dates=['date'])
    stores = pd.read_csv('data/stores.csv')
    oil = pd.read_csv('data/oil.csv', parse_dates=['date'])
    holidays = pd.read_csv('data/holidays_events.csv', parse_dates=['date'])
    
    # Simple merge logic within the task
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
    
    def merge_logic(df):
        df = df.merge(stores, on='store_nbr', how='left')
        df = df.merge(oil, on='date', how='left')
        return df

    return merge_logic(train), merge_logic(test)

@task
def transform_data(train, test):
    # Combine to ensure feature consistency (especially for lags)
    full_df = pd.concat([train, test], axis=0).reset_index(drop=True)
    full_df = add_target_encoding(full_df)
    full_df = create_features(full_df)
    
    train_f = full_df[full_df['sales'].notnull()]
    test_f = full_df[full_df['sales'].isnull()]
    return train_f, test_f

@task
def train_parallel(train_f, test_f, config):
    if not ray.is_initialized():
        ray.init()
    
    families = train_f['family'].unique()
    features = [
        'store_nbr', 'onpromotion', 'dcoilwtico', 
        'day_sin', 'day_cos', 'weeks_since_earthquake', 
        'lag_16', 'lag_21', 'lag_364',
        'rolling_mean_14', 'store_fam_avg_sales',
        'is_payday', 'days_to_payday',
        'oil_7d_avg', 'oil_30d_avg',
    ]
    
    futures = []
    for fam in families:
        train_fam = train_f[train_f['family'] == fam]
        test_fam = test_f[test_f['family'] == fam]
        
        futures.append(
            train_family_model.remote(fam, train_fam, test_fam, features, config)
        )
    
    results = ray.get(futures)
    ray.shutdown()
    return pd.concat(results).sort_values('id')

@flow(name="Store Sales Parameter Search")
def store_sales_flow(config, run_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Kaggle_Store_Sales")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        
        train, test = load_data()
        train_f, test_f = transform_data(train, test)
        
        submission = train_parallel(train_f, test_f, config)
        
        file_path = f'submissions/submission_{run_name}.csv'
        submission.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        best_lr = float(sys.argv[1])
        best_leaves = int(sys.argv[2])
        best_est = int(sys.argv[3])
        
        manual_config = {
            "learning_rate": best_lr,
            "num_leaves": best_leaves,
            "n_estimators": best_est
        }
        
        run_name = f"manual_best_lr{best_lr}_nl{best_leaves}"
        print(f"🚀 Running Manual Best Config: {run_name}")
        store_sales_flow(manual_config, run_name)
        
    else:
        test_configs = [
            {
                "name": "baseline_lgb",
                "params": {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 200}
            },
            {
                "name": "deeper_trees",
                "params": {"learning_rate": 0.03, "num_leaves": 64, "n_estimators": 300}
            }
        ]

        for test in test_configs:
            print(f"Running experiment: {test['name']}")
            store_sales_flow(test['params'], test['name'])