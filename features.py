import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
from tqdm import tqdm
# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']

def make_time_series_data(Input, stand):
    
    # 기준을 잡습니다. 기준은 여기서 %Y-%m 입니다.
    standard = ['customer_id'] + [stand]
    data = Input.copy()    
    
    
    data[stand] = pd.to_datetime(data.order_date).dt.strftime(stand)    
    data.order_date = pd.to_datetime(data.order_date)
    
    # 월단위의 틀을 만들어주고, stand으로 aggregation을 해준 다음에 merge를 해줄 것입니다
    times = pd.date_range('2009-12-01', periods= (data.order_date.max() - data.order_date.min()).days + 1, freq='1d')    
    customerid_frame = np.repeat(data.customer_id.unique(), len(times))
    date_frame = np.tile(times, len(data.customer_id.unique()))

    frame = pd.DataFrame({'customer_id':customerid_frame,'order_date':date_frame})
    frame[stand] = pd.to_datetime(frame.order_date).dt.strftime(stand)
    
    # group by
    data_group = data.groupby(standard).sum().reset_index()
    frame_group = frame.groupby(standard).count().reset_index().drop(['order_date'], axis=1)
    
    # merge
    merge = pd.merge(frame_group, data_group, on=standard, how='left').fillna(0)
    merge = merge.rename(columns={stand : 'standard'})
    return merge
'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''
def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    
    # cust = df[df['year_month']<=year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label


def feature_preprocessing(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te

def set_datetime(year_month,months):
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    year_month = d - dateutil.relativedelta.relativedelta(months=months)
    year_month = year_month.strftime('%Y-%m')
    return year_month

def custom_features(merge_data,orig_data,year_month):
    df=merge_data.copy()
    previous_over300=[]

    mean_consusm=[]
    mean_consusm_no_refund=[]

    # quater
    compare_list=[]
    last_year=df[(df['standard']>set_datetime(year_month,16)) & (df['standard']<set_datetime(year_month,11))]
    current_year=df[df['standard']>set_datetime(year_month,4)]
    ###

    # compare with 12
    ts_prev=[]
    ts_real_prev=[]
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_1y = d - dateutil.relativedelta.relativedelta(years=1)
    prev_1y = prev_1y.strftime('%Y-%m')
    prev_2y = d - dateutil.relativedelta.relativedelta(years=2)
    prev_2y = prev_2y.strftime('%Y-%m')
    ###
    post_feature=[]
    for customer in tqdm(df.customer_id.unique()):
        transaction=df[df.customer_id==customer]
        real_transaction=transaction[transaction.total>0]
        # print(real_transaction)
        # print(len(real_transaction))
        #### over_300 ratio
        real_transaction_count=len(real_transaction)
        transaction_over300_count=len(real_transaction[real_transaction.total>300])

        if real_transaction_count==0:
            previous_over300.append(0)
        else:
            previous_over300.append(transaction_over300_count/real_transaction_count)
        ####

        #### quater_compare
        last_temp=last_year[last_year.customer_id==customer]
        current_temp=current_year[current_year.customer_id==customer]

        last_over300=last_temp[last_temp.total>300].count().total.item()
        current_over300=current_temp[current_temp.total>300].count().total.item()
        compare_list.append((1-(current_over300-last_over300+1)/4)*last_over300)
        ####

        ### compare with 12
        
        ts_1=transaction[transaction['standard']==prev_1y].total.to_numpy()
        ts_2=transaction[transaction['standard']==prev_2y].total.to_numpy()

        real_ts_1=real_transaction[real_transaction['standard']==prev_1y].total.to_numpy()
        real_ts_2=real_transaction[real_transaction['standard']==prev_2y].total.to_numpy()

        if len(ts_1)==0:
            ts_1=0
        else:
            ts_1=ts_1[0]
        if len(ts_2)==0:
            ts_2=0
        else:
            ts_2=ts_2[0]

        if len(real_ts_1)==0:
            real_ts_1=0
        else:
            real_ts_1=real_ts_1[0]
        if len(real_ts_2)==0:
            real_ts_2=0
        else:
            real_ts_2=real_ts_2[0]
        ts=(ts_1+ts_2)/2
        real_ts=(real_ts_1+real_ts_2)/2
        ts_prev.append(ts)
        ts_real_prev.append(real_ts)
        ####

        # ### post feature

        
        post_y=post_n=over_300=under300=0
        post_ts=orig_data[(orig_data['customer_id']==customer)&(orig_data['product_id']=='POST')]
        post_ts=post_ts[post_ts.total>0]
        cnt_sum,cnt1_sum=0,0
        
        for yr_m_post in post_ts.order_date.dt.strftime('%Y-%m').unique():
            get_customer_info=df[(df['customer_id']==customer)&(df['standard']==yr_m_post)]
            cnt=get_customer_info[get_customer_info['total']>=300].count()    
            cnt1=get_customer_info[(get_customer_info['total']<300)&(get_customer_info['total']>0)].count()    

            cnt_sum+=cnt.values[0]
            cnt1_sum+=cnt1.values[0]
        if cnt1_sum+cnt_sum==0:
            post_n+=1
            post_feature.append(0)
        else:
            post_y+=1
            post_feature.append(cnt_sum-cnt1_sum)
        ### cunsum_mean
        try:
            first_transaction=real_transaction.iloc[0].standard
        except:
            mean_consusm.append(-1000)
            mean_consusm_no_refund.append(-1000)
            continue
        c_d=datetime.datetime.strptime(year_month, "%Y-%m")     
        d = datetime.datetime.strptime(first_transaction, "%Y-%m")
        use_period=(c_d-d).days/(365.25/12)
        mean_consusm.append(transaction.total.sum()/use_period)
        mean_consusm_no_refund.append(sum(transaction.total[transaction.total>0])/use_period)
        ####

    #####
    # mean_consusm=np.array(mean_consusm)
    # mean_consusm_no_refund=np.array(mean_consusm_no_refund)
    # return previous_over300,compare_list,np.where(mean_consusm>300,0,np.where(mean_consusm>600,2,1)),np.where(mean_consusm_no_refund>300,0,np.where(mean_consusm_no_refund>600,2,1))
    return previous_over300,compare_list,ts_prev,ts_real_prev,mean_consusm,mean_consusm_no_refund,post_feature
def feature_engineering1(df, year_month):
    df = df.copy()
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    # prev_ym = d - dateutil.relativedelta.relativedelta(years=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    # train, test 데이터 선택
    # print(prev_ym)
    train = df[df['order_date'] <prev_ym]
    test = df[df['order_date'] < year_month]
    # print(train.shape)
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']] 
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    print(train_label.shape)
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_func)

        # 멀티 레벨 컬럼을 사용하기 쉽게 1 레벨 컬럼명으로 변경
        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    # print(train)
    # print(train_label)
    # print(all_train_data.shape)
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    merge_train=make_time_series_data(train,'%Y-%m')
    merge_test=make_time_series_data(test,'%Y-%m')
    print('merge_done')
    # print(merge_train)
    # exit()
    # print(merge_train.shape)
    train_previous_over300,train_compare_list,train_mean_consusm,train_mean_consusm_no_refund=custom_features(merge_train,prev_ym)
    all_train_data['compare_quater']=train_previous_over300
    all_train_data['ratio_over300']=train_compare_list
    all_train_data['mean_consusm']=train_mean_consusm
    all_train_data['mean_consusm_no_refund']=train_mean_consusm_no_refund
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns

    # print(features)
    # print(all_train_data.head())
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_func)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')
    test_previous_over300,test_compare_list,test_mean_consusm,test_mean_consusm_no_refund=custom_features(merge_test,year_month)
    test_data['compare_quater']=test_previous_over300
    test_data['ratio_over300']=test_compare_list
    test_data['mean_consusm']=test_mean_consusm
    test_data['mean_consusm_no_refund']=test_mean_consusm_no_refund
    # print(all_train_data)
    # print(test_data)
    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    print(all_train_data)
    print(x_tr)
    print(x_te)
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features
def get_refund(is_refund, total):
    if is_refund:
        return -total
    return 0

def feature_engineering2(df, year_month):
    df = df.copy()
    df['month']=df['order_date'].dt.month
    df['year_month']=df['order_date'].dt.strftime('%Y-%m')
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()    
    
    # order_ts
    df['order_ts']=df['order_date'].astype(np.int64)//1e9
    df['order_ts_plus'] = (df[df['total'] > 0]['order_date'].astype(np.int64) // 1e9)
    df['order_ts_plus']=df['order_ts_plus'].fillna(0)
    df['order_ts_diff']=df.groupby(['customer_id'])['order_ts'].diff()
    df['quantity_diff']=df.groupby(['customer_id'])['quantity'].diff()
    df['price_diff']=df.groupby(['customer_id'])['price'].diff()
    df['total_diff']=df.groupby(['customer_id'])['total'].diff()

    # 환불
    df['is_refund'] = df.order_id.str.contains('C')
    df['refund_total'] = df.apply(lambda x:get_refund(x['is_refund'], x['total']), axis=1)
    df['refund_quantity'] = df.apply(lambda x:get_refund(x['is_refund'], x['quantity']), axis=1)
    df['refund_total_cumsum']=df.groupby(['customer_id'])['refund_total'].cumsum()   
    df['refund_quantity_cumsum']=df.groupby(['customer_id'])['refund_quantity'].cumsum()   
    df['refund_ratio']=df.groupby(['customer_id'])['refund_total'].cumsum() /df.groupby(['customer_id'])['total'].cumsum() 
    # print(df)
    # exit() 
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    # 2011-10 2011-11
    prev_temp=prev_ym
    # prev_ym=year_month

    # print(prev_temp)
    # print(prev_ym)
    # print(year_month)
    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    
    # train, test 레이블 데이터 생성
    train_label = generate_label(df, prev_temp)[['customer_id','year_month','label']]
    test_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    # print(train_label)
    # print(test_label)



    # group by aggregation 함수 선언
    # agg_func = ['mean','max','min','sum','count','std','skew']
    agg_func = ['mean','sum','std','skew']
    agg_func1 = ['max','mean','sum','std','skew']
    agg_dict = {
        'order_ts' :['first','last'],
        'order_ts_plus' :['first','last'],
        'order_ts_diff' :agg_func1,
        'quantity_diff' :agg_func1,
        'price_diff' :agg_func,
        'total_diff' :agg_func1,
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'is_refund': ['sum','skew'],
        'refund_total': ['sum','skew'],
        'refund_quantity': ['sum','skew'],
        'refund_total_cumsum': agg_func,
        'refund_quantity_cumsum': agg_func,
        'refund_quantity_cumsum': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
    }
    # print(df)
    # exit()
    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)
        # print(agg_dict)
        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')

    
    # train
    print('merge_start')
    merge_train=make_time_series_data(train,'%Y-%m')
    merge_test=make_time_series_data(test,'%Y-%m')
    print('merge_done')
    train_over_300,train_compare_list,train_prev_12,train_real_prev_12,train_mean_spend,train_mean_no_spend,train_postF=custom_features(merge_train,train,prev_ym)
    test_over_300,test_compare_list,test_prev_12,test_real_prev_12,test_mean_spend,test_mean_no_spend,test_postF=custom_features(merge_test,test,year_month)

    all_train_data['over_300_ratio']=train_over_300
    all_train_data['compare_quater']=train_compare_list
    all_train_data['compare_12']=train_prev_12
    all_train_data['compare_real_12']=train_real_prev_12
    all_train_data['mean_spend']=train_mean_spend
    all_train_data['mean_no_spend']=train_mean_no_spend
    all_train_data['postF']=train_postF
    # print(all_train_data)
    train_df=train
    cols=['month','year_month']
    train_df_agg=train_df.groupby(['customer_id'])[cols].agg([lambda x:x.value_counts().index[0]])
    train_df_agg.columns=['month-mode','year_moth-mode']
    all_train_data=train_df_agg.merge(all_train_data,on=['customer_id'],how='left')
    print(all_train_data)
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    test_df=test
    cols=['month','year_month']
    test_df_agg=test_df.groupby(['customer_id'])[cols].agg([lambda x:x.value_counts().index[0]])
    test_df_agg.columns=['month-mode','year_moth-mode']
    test_data=test_df_agg.merge(test_data,on=['customer_id'],how='left')
    # print(test_data)
    test_data['over_300_ratio']=test_over_300
    test_data['compare_quater']=test_compare_list 
    test_data['compare_12']=test_prev_12     
    test_data['compare_real_12']=test_real_prev_12     
    test_data['mean_spend']=test_mean_spend
    test_data['mean_no_spend']=test_mean_no_spend
    test_data['postF']=test_postF

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing(all_train_data, test_data, features)
    print(all_train_data)
    print(x_tr)
    print(x_te)
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features
if __name__ == '__main__':
    data_dir='/opt/ml/code/input'
    print('data_dir', data_dir)
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    feature_engineering2(data,'2011-12')