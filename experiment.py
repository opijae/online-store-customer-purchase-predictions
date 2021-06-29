#%%
import pandas as pd
import numpy as np
import features
import inference
# %%
data=pd.read_csv('../input/train.csv', parse_dates=['order_date'])
#%%
# 환불을 제외한 데이터
data_pos=data[data['total']>0]
#%%
#이상치 제거
remove_data=inference.remove_sepcial(data)
#%%
train, test, y, feature=features.feature_engineering2(data,'2011-12')
#%%
train_pos, test_pos, y, feature=features.feature_engineering2(data_pos,'2011-12')
#%%
train_remove, test_remove, y_remove, feature_remove=features.feature_engineering2(remove_data,'2011-12')
#525,1118
#%%
path_smooth=10000
min_split_gain=0.7
model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 100,
            'seed': 42,
            'verbose': -1,
            'n_jobs': -1,
            # 'patient': 30,
            # 'path_smooth': path_smooth,
            # 'min_split_gain' : min_split_gain
        }
#%%  quantile scaling
from sklearn.preprocessing import QuantileTransformer
# df=train.copy()
qt=QuantileTransformer(n_quantiles=1000, random_state=42)
train_qt=qt.fit_transform(train[feature])
train_qt = pd.DataFrame(train_qt,  columns=train[feature].columns)
test_qt=qt.fit_transform(test[feature])
test_qt = pd.DataFrame(test_qt,  columns=test[feature].columns)
# %%    pca
from sklearn.decomposition import PCA

pca = PCA(n_components = 0.90) 
# pca.fit(X)
# pc_score = pca.transform(X) # array 형태로 return
train_pca=pca.fit_transform(train_qt[feature])
test_pca=pca.fit_transform(test_qt[feature])

train_pca_df=pd.DataFrame(train_pca,columns=np.arange(22))
test_pca_df=pd.DataFrame(test_pca,columns=np.arange(22))

train_pca=pd.concat([train_pca_df,train],axis=1)
test_pca=pd.concat([test_pca_df,test],axis=1)
feature_pca=feature.append(train_pca_df.columns)
# %% 추론
experiment_name='all_addrefund_qt1000_pca'
import mlflow
mlflow.end_run()
y_oof, test_preds, fi = inference.make_lgb_oof_prediction(train_pca, y, test_pca, feature_pca,args=f'{experiment_name}' ,model_params=model_params)
# %%
fi.to_csv(f'{experiment_name}.csv',sep=',')

#%%
merge_data=features.make_time_series_data(data,'%Y-%m')
# %%
##### tab net

from pytorch_tabnet.tab_model import TabNetClassifier,TabNetRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.metrics import Metric

experiment_name='tabnet_30'

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
n_train=train[feature].to_numpy()
n_y=y.to_numpy()
n_test=test[feature].to_numpy()
test_preds = np.ones(n_test.shape[0])
y_oof = np.zeros(y.shape[0])
score=0
class Gini(Metric):
    def __init__(self):
        self._name = "gini"
        self._maximize = True

    def __call__(self, y_true, y_score):
        auc = roc_auc_score(y_true, y_score[:, 1])
        return max(2*auc - 1, 0.)
        # return auc
for fold,(tr_idx,val_idx) in enumerate(skf.split(train,y)):
    # x_tr, x_val = train.loc[tr_idx, feature], train.loc[val_idx, feature]

    clf=TabNetClassifier()
    x_tr,x_val=n_train[tr_idx],n_train[val_idx]
    y_tr,y_val=n_y[tr_idx],n_y[val_idx]

    clf.fit(
        X_train=x_tr,
        y_train=y_tr,
        eval_set=[(x_val,y_val)],
        eval_metric=[Gini]
    )
    val_preds=clf.predict_proba(x_val)[:,1]

    y_oof[val_idx]=val_preds

    print(f'Fold {fold+1} | AUC {roc_auc_score(y_val,val_preds)}')
    print('-'*80)

    score+=(roc_auc_score(y_val,val_preds)/10)

    test_preds *= (clf.predict_proba(n_test)[:,1]+np.finfo(float).eps)
    test_preds=np.float64(test_preds)

    del x_tr, x_val, y_tr, y_val
test_preds=test_preds**(1/10)
print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력
# %%
sub = pd.read_csv( '../input/sample_submission.csv')
sub['probability'] = test_preds
sub.to_csv(f'{experiment_name}.csv', index=False)

