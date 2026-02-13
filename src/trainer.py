import ray
import lightgbm as lgb
import numpy as np
import pandas as pd # type: ignore

@ray.remote
def train_family_model(fam, train_fam, test_fam, features):
    """
    This function runs independently for each product family.
    Ray will run multiple instances of this at once.
    """
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
    }

    dtrain = lgb.Dataset(train_fam[features], label=np.log1p(train_fam['sales']))
    
    model = lgb.train(lgb_params, dtrain, num_boost_round=200)
    preds = np.expm1(model.predict(test_fam[features]))
    
    return pd.DataFrame({'id': test_fam['id'], 'sales': preds})