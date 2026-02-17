import mlflow
import ray
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore

@ray.remote
def train_family_model(fam, train_fam, test_fam, features, params):
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    
    split_date = train_fam['date'].max() - pd.Timedelta(days=15)
    actual_train = train_fam[train_fam['date'] <= split_date].copy()
    local_val = train_fam[train_fam['date'] > split_date].copy()

    actual_train['is_selling'] = (actual_train['sales'] > 0).astype(int)
    threshold = params.get('zero_threshold', 0.1)

    with mlflow.start_run(run_name=f"Family_{fam}", nested=True):
        clf_params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1}
        dtrain_clf = lgb.Dataset(actual_train[features], label=actual_train['is_selling'])
        clf_model = lgb.train(clf_params, dtrain_clf, num_boost_round=100)
        
        lgb_params = params.copy()
        n_rounds = lgb_params.pop('n_estimators', 200) 
        lgb_params.update({'objective': 'regression', 'metric': 'rmse', 'verbosity': -1})
        
        dtrain_lgb = lgb.Dataset(actual_train[features], label=np.log1p(actual_train['sales']))
        lgb_model = lgb.train(lgb_params, dtrain_lgb, num_boost_round=n_rounds)

        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': lgb_params.get('learning_rate', 0.05),
            'max_depth': 6,
            'nthread': 1 
        }
        
        dtrain_xgb = xgb.DMatrix(actual_train[features], label=np.log1p(actual_train['sales']))
        dtest_xgb = xgb.DMatrix(test_fam[features])
        dval_xgb = xgb.DMatrix(local_val[features])
        
        xgb_model = xgb.train(xgb_params, dtrain_xgb, num_boost_round=n_rounds)

        lgb_test_preds = np.expm1(lgb_model.predict(test_fam[features]))
        xgb_test_preds = np.expm1(xgb_model.predict(dtest_xgb))
        
        lgb_val_preds = np.expm1(lgb_model.predict(local_val[features]))
        xgb_val_preds = np.expm1(xgb_model.predict(dval_xgb))

        blended_test_preds = (0.7 * lgb_test_preds) + (0.3 * xgb_test_preds)
        blended_val_preds = (0.7 * lgb_val_preds) + (0.3 * xgb_val_preds)

        test_clf_probs = clf_model.predict(test_fam[features])
        val_clf_probs = clf_model.predict(local_val[features])
        
        final_test_preds = np.where(test_clf_probs < threshold, 0, blended_test_preds)
        final_val_preds = np.where(val_clf_probs < threshold, 0, blended_val_preds)

        rmsle = np.sqrt(mean_squared_error(np.log1p(local_val['sales']), np.log1p(final_val_preds)))
        mlflow.log_metric("local_rmsle", rmsle)
        
        return pd.DataFrame({'id': test_fam['id'], 'sales': final_test_preds})