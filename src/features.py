import numpy as np
import pandas as pd # type: ignore

def create_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    earthquake_date = pd.to_datetime('2016-04-16')
    df['weeks_since_earthquake'] = (df['date'] - earthquake_date).dt.days // 7
    df['weeks_since_earthquake'] = df['weeks_since_earthquake'].clip(lower=0) 

    df['day_of_month'] = df['date'].dt.day
    df['is_payday'] = ((df['day_of_month'] == 15) | (df.date.dt.is_month_end)).astype(int)
    
    def days_to_payday(date):
        if date.day <= 15:
            return 15 - date.day
        else:
            return (date + pd.offsets.MonthEnd(0)).day - date.day
            
    df['days_to_payday'] = df['date'].apply(days_to_payday)

    df['oil_7d_avg'] = df['dcoilwtico'].rolling(7).mean()
    df['oil_30d_avg'] = df['dcoilwtico'].rolling(30).mean()

    df['lag_16'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(16))
    df['rolling_mean_14'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(14).mean())
    df['lag_364'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(364))
    df['lag_21'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(21))
    return df


def add_target_encoding(df):
    """
    Computes the average sales for each store and family 
    based on the training data.
    """
    encoding = df[df['sales'].notnull()].groupby(['store_nbr', 'family'])['sales'].mean().reset_index()
    encoding.rename(columns={'sales': 'store_fam_avg_sales'}, inplace=True)
    
    df = df.merge(encoding, on=['store_nbr', 'family'], how='left')
    
    df['store_fam_avg_sales'] = df['store_fam_avg_sales'].fillna(df['sales'].mean())
    return df