import pandas as pd # type: ignore
import numpy as np
import ray
from prefect import task, flow
from src.features import create_features
from src.trainer import train_family_model

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
    full_df = create_features(full_df)
    
    train_f = full_df[full_df['sales'].notnull()]
    test_f = full_df[full_df['sales'].isnull()]
    return train_f, test_f

@task
def train_parallel(train_f, test_f):
    if not ray.is_initialized():
        ray.init()
    
    families = train_f['family'].unique()
    features = ['store_nbr', 'onpromotion', 'dcoilwtico', 'day_sin', 
                'day_cos', 'weeks_since_earthquake', 'lag_16', 'rolling_mean_14']
    
    train_ref = ray.put(train_f)
    test_ref = ray.put(test_f)
    
    futures = []
    for fam in families:
        train_fam = train_f[train_f['family'] == fam]
        test_fam = test_f[test_f['family'] == fam]
        
        futures.append(
            train_family_model.remote(fam, train_fam, test_fam, features)
        )
    
    results = ray.get(futures)
    
    ray.shutdown()
    return pd.concat(results).sort_values('id')

@flow(name="Store Sales Forecasting Pipeline")
def store_sales_flow():
    train, test = load_data()
    train_f, test_f = transform_data(train, test)
    
    submission = train_parallel(train_f, test_f)
    submission.to_csv('submissions/ray_prefect_submission.csv', index=False)

if __name__ == "__main__":
    store_sales_flow()