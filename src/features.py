import numpy as np
import pandas as pd # type: ignore

def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    earthquake_date = pd.to_datetime('2016-04-16')
    df['weeks_since_earthquake'] = (df['date'] - earthquake_date).dt.days // 7
    df['weeks_since_earthquake'] = df['weeks_since_earthquake'].clip(lower=0) 

    df['lag_16'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(16))
    df['rolling_mean_14'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14).mean())
    return df