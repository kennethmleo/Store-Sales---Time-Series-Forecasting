import mlflow
import ray
import lightgbm as lgb
import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

@ray.remote
def train_family_model(fam, train_fam, test_fam, features, params):
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    
    lgb_params = params.copy()
    n_rounds = lgb_params.pop('n_estimators', 200) 
    
    lgb_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1
    })

    split_date = train_fam['date'].max() - pd.Timedelta(days=15)
    
    actual_train = train_fam[train_fam['date'] <= split_date]
    local_val = train_fam[train_fam['date'] > split_date]

    with mlflow.start_run(run_name=f"Family_{fam}", nested=True):
        mlflow.log_params(lgb_params)
        mlflow.log_param("n_estimators", n_rounds)
        
        dtrain = lgb.Dataset(actual_train[features], label=np.log1p(actual_train['sales']))
        dval = lgb.Dataset(local_val[features], label=np.log1p(local_val['sales']), reference=dtrain)
        
        model = lgb.train(
            lgb_params, 
            dtrain, 
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=20)]
        )
        
        val_preds = model.predict(local_val[features])
        rmsle = np.sqrt(mean_squared_error(np.log1p(local_val['sales']), val_preds))
        mlflow.log_metric("local_rmsle", rmsle)
        
        test_preds = np.expm1(model.predict(test_fam[features]))
        
        return pd.DataFrame({'id': test_fam['id'], 'sales': test_preds})