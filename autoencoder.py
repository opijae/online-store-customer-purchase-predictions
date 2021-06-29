#%%
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.arrays import categorical

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.std import tqdm
from torch.autograd import Variable

from data_preparation import data_to_image
from model import model_res
from train import train_model

torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

%load_ext autoreload
%autoreload 2
%matplotlib inline
warnings.filterwarnings("ignore")
%config InlineBackend.figure_format = 'retina'
# %%
data=pd.read_csv('../../input/train.csv', parse_dates=["order_date"])
# %%
import sys
sys.path.insert(1,'/opt/ml/code/src')
import features
train,test,y,feature=merge_data=features.feature_engineering2(data,'2011-12')

#%%
train=train[feature].to_numpy()
test=test[feature].to_numpy()

# %%
x_train, x_val, y_train, y_val =train_test_split(train,y)
y_train=y_train.to_numpy()
y_val=y_val.to_numpy()

# %%
class Autoencoder(nn.Module):
    def __init__(self,input_shape,encoding_dim):
        super(Autoencoder, self).__init__()

        self.encode=nn.Sequential(
            nn.Linear(input_shape,50),
            nn.ReLU(True),
            nn.Linear(50,encoding_dim),
            # nn.ReLU(True),
            # nn.Linear(128,64),
            # nn.ReLU(True),
            # nn.Linear(64,encoding_dim),
            # nn.Sigmoid(),
        )
        # self.sig=nn.Sigmoid()
        self.decode=nn.Sequential(
            nn.Linear(encoding_dim,50),
            nn.ReLU(True),
            # nn.Linear(50,100),
            # nn.ReLU(True),
            nn.Linear(50,input_shape),
            # nn.ReLU(True),
            # nn.Linear(256,input_shape),
            
        )
    def forward(self,x):
        x=self.encode(x)
        x=self.decode(x)
        return x

#%%
x_train_t = torch.from_numpy(x_train).float()
y_train_t = torch.from_numpy(y_train).long()
x_val_t = torch.from_numpy(x_val).float()
y_val_t = torch.from_numpy(y_val).long()
train_dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(x_val_t, y_val_t)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)

# %%
import torch.optim as optim

autoencoder=Autoencoder(train.shape[1],25).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0003)
# %%
train_loss_list=[]
val_loss_list=[]
epochs=1000
best_loss=np.inf
for epoch in range(epochs):
    autoencoder.train()
    running_loss=0.0
    for i, data in enumerate(train_loader):
        inputs,label=data
        inputs=inputs.to(device)
        # label=label.to(device)
        optimizer.zero_grad()

        outputs=autoencoder(inputs)
        # 128
        loss=criterion(outputs,inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if (epoch+1) %50==0:
        #     # print(inputs)
        #     # print(outputs)
        #     # break
        # print(f'epochs : {epoch+1}, step : {i}/ {int(len(train_dataset)/256)},loss={running_loss/(i+1)}',end='\r')
    train_loss_list.append(running_loss/len(train_dataset))
    print()
    val_loss=0
    with torch.no_grad():
        autoencoder.eval()
        for data in val_loader:
            inputs,label=data
            # print(data)
            # print(inputs)
            inputs=inputs.to(device)
            label=label.to(device)            
            outputs=autoencoder(inputs)
            loss=criterion(outputs,inputs)
            val_loss+=loss.item()
        print(f'valid  epochs : {epoch+1}, loss={val_loss/(len(val_dataset))}',end='\r')
        if best_loss>(val_loss/(len(val_dataset))):
            # print('loss')
            best_loss=val_loss/(len(val_dataset))
    
    val_loss_list.append(running_loss/len(val_dataset))
    print()
    print()
print(best_loss)

#%%
# 오토 인코더 피처 뽑기
test_t = torch.from_numpy(test).float()
test_dataset = torch.utils.data.TensorDataset(test_t)
test_loader = torch.utils.data.DataLoader(test_dataset)

new_test=[]

for inputs in test_loader:
    # print(inputs)
    inputs=inputs[0].to(device)
    with torch.no_grad():
        outputs=autoencoder(inputs)
    # print(outputs)
    # print('-'*80)
    pred=outputs.cpu().numpy()
    new_test.extend(pred)
new_test=np.array(new_test)
#%%
# new_test=np.array(new_test)
new_train=[]
train_orig_t = torch.from_numpy(train).float()
train_orig_dataset = torch.utils.data.TensorDataset(train_orig_t)
train_orig_loader = torch.utils.data.DataLoader(train_orig_dataset)

for inputs in train_orig_loader:
    # print(inputs)
    inputs=inputs[0].to(device)
    with torch.no_grad():
        outputs=autoencoder(inputs)
    # print(outputs)
    # print('-'*80)
    pred=outputs.cpu().numpy()
    new_train.extend(pred)
new_train=np.array(new_train)
#%%
new_train
# %%
# 성능 측정
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 100, # 트리 개수
            'seed': 42,
            'verbose': -1,
            'n_jobs': -1,    
        }
y_oof = np.zeros(new_train.shape[0])
test_preds = np.zeros(new_test.shape[0])
y_n=y.to_numpy()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold,(tr_idx,val_idx) in enumerate(skf.split(new_train,y_n)):
    x_tr,x_val=new_train[tr_idx],new_train[val_idx]
    y_tr,y_val=y_n[tr_idx],y_n[val_idx]

    dtrain=lgb.Dataset(x_tr,label=y_tr)
    dvalid=lgb.Dataset(x_val,label=y_val)

    clf=lgb.train(
        model_params,
        dtrain,
        valid_sets=[dtrain,dvalid],
        categorical_feature='auto',
        verbose_eval=200
    )

    val_preds=clf.predict(x_val)

    y_oof[val_idx] = val_preds
        
    # 폴드별 Validation 스코어 측정
    print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
    print('-'*80)
    test_preds+=clf.predict(test) / (fold+1)
# %%
temp=test_preds

# %%
len(new_train)
# %%
with open("../../input/sample_submission.csv", "r") as f:
        f.readline()
        id_list = [line.strip().split(",")[0] for line in f]
with open("submissions.csv", "w") as f:
    f.write("customer_id,probability\n")
    for test_id, prediction in zip(id_list, test_preds.ravel()):
        f.write(f"{test_id},{prediction}\n")
# %%
