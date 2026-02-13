import mlflow
import ray
import lightgbm as lgb
import numpy as np
import pandas as pd # type: ignore

@ray.remote
def train_family_model(fam, train_fam, test_fam, features, params):

    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    
    # 2. Extract n_estimators before passing params to LightGBM
    # We copy to avoid modifying the original dictionary for other workers
    lgb_params = params.copy()
    n_rounds = lgb_params.pop('n_estimators', 200) 
    
    # 3. Add standard requirements
    lgb_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1
    })

    with mlflow.start_run(run_name=f"Family_{fam}", nested=True):
        mlflow.log_params(lgb_params)
        mlflow.log_param("n_estimators", n_rounds)
        
        # Prepare dataset
        dtrain = lgb.Dataset(train_fam[features], label=np.log1p(train_fam['sales']))
        
        # Train
        model = lgb.train(
            lgb_params, 
            dtrain, 
            num_boost_round=n_rounds
        )
        
        # Predict
        preds = np.expm1(model.predict(test_fam[features]))
        
        # Optional: If you had validation data, you could log metrics here
        # mlflow.log_metric("rmse", model.best_score['training']['rmse'])
        
        return pd.DataFrame({'id': test_fam['id'], 'sales': preds})