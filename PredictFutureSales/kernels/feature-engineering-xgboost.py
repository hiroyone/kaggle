# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"_uuid": "55f0ed2ac468f0eb59c24726ca0c9bdc8884c345", "cell_type": "markdown"}
# This notebook is simpified version of the final project in the [How to Win a Data Science Competition: Learn from Top Kagglers](https://www.coursera.org/learn/competitive-data-science) course. Simplified means without ensembling.
#
# #### Pipline
# * load data
# * heal data and remove outliers
# * work with shops/items/cats objects and features
# * create matrix as product of item/shop pairs within each month in the train set
# * get monthly sales for each item/shop pair in the train set and merge it to the matrix
# * clip item_cnt_month by (0,20)
# * append test to the matrix, fill 34 month nans with zeros
# * merge shops/items/cats to the matrix
# * add target lag features
# * add mean encoded features
# * add price trend features
# * add month
# * add days
# * add months since last sale/months since first sale features
# * cut first year and drop columns which can not be calculated for the test set
# * select best features
# * set validation strategy 34 test, 33 validation, less than 33 train
# * fit the model, predict and clip targets for the test set

# + {"_uuid": "f03379ee467570732ebb2b3d20062fea0584d57d", "cell_type": "markdown"}
# # Part 1, perfect features

# + {"_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"}
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle
sys.version_info

# + {"_uuid": "f0a1c729d4fb3d6609f9dfb163ebe92fa9dc654c"}
items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cats = pd.read_csv('../input/item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')
# set index to ID to avoid droping it later
test  = pd.read_csv('../input/test.csv').set_index('ID')

# + {"_uuid": "ed7a190645750a818e29a6291ba2553a91764c7c", "cell_type": "markdown"}
# ## Outliers

# + {"_uuid": "425d8f2dc08378977b393bf80c5fdcf0fba2c992", "cell_type": "markdown"}
# There are items with strange prices and sales. After detailed exploration I decided to remove items with price > 100000 and sales > 1001 (1000 is ok).

# + {"_uuid": "5a864412fafc3129a3e9bd5bb1f18a7cf0c62935"}
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price)

# + {"_uuid": "7e621535d112603c60aeb2c2f83dbbf96d36b732"}
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
# -

# item price < 0
train[train.item_price <0]

# + {"_uuid": "d2f99368478e3063b1c379537944e954d7186928", "cell_type": "markdown"}
# There is one item with price below zero. Fill it with median.

# + {"_uuid": "0fc6b90b22fe232f4240ac8f965cc52b3db5526a"}
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median

# + {"_uuid": "7da194c285d696b5c6978148bf0143b9b2a7b0c5", "cell_type": "markdown"}
# Several shops are duplicates of each other (according to its name). Fix train and test set.
# -

shops.sort_values(by='shop_name' )

# + {"_uuid": "00fe91e9c482ea413abd774ff903fe3d152785dd"}
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

# + {"_uuid": "a30f0521464e1fa20444e66d24bbdcb76b93f6de", "cell_type": "markdown"}
# ## Shops/Cats/Items preprocessing
# Observations:
# * Each shop_name starts with the city name.
# * Each category contains type and subtype in its name.

# + {"_uuid": "12fae4c8d0c8f3e817307d1e0ffc6831e9a8d696"}
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','type_code', 'subtype_code']]

items.drop(['item_name'], axis=1, inplace=True)
# -

shops.head()

items.head()

cats.head()

# + {"_uuid": "62c5f83fa222595da99294f465ab28e80ce415e9", "cell_type": "markdown"}
# ## Monthly sales
# Test set is a product of some shops and some items within 34 month. There are 5100 items * 42 shops = 214200 pairs. 363 items are new compared to the train. Hence, for the most of the items in the test set target value should be zero. 
# In the other hand train set contains only pairs which were sold or returned in the past. Tha main idea is to calculate monthly sales and <b>extend it with zero sales</b> for each unique pair within the month. This way train data will be similar to test data.

# + {"_uuid": "fb69350aef2c28cdb619e2532de1e24ab3c43899"}
len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)

# + {"_uuid": "7626c7455ea71b65894c6c866519df15080fa2ac"}
ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
time.time() - ts
# -

matrix.head(n=20)

# + {"_uuid": "867e91a7570dd78b4834f4f1a166e58f80b63f93", "cell_type": "markdown"}
# Aggregate train set by shop/item pairs to calculate target aggreagates, then <b>clip(0,20)</b> target value. This way train target will be similar to the test predictions.
#
# <i>I use floats instead of ints for item_cnt_month to avoid downcasting it after concatination with the test set later. If it would be int16, after concatination with NaN values it becomes int64, but foat16 becomes float16 even with NaNs.</i>

# + {"_uuid": "9fef5477060be7d2e6c85dcb79d8e18e6253f7dd"}
train['revenue'] = train['item_price'] *  train['item_cnt_day']

# + {"_uuid": "7dd27181918fc7df89676e24d72130d183929d2d"}
ts = time.time()
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))
time.time() - ts
# -

matrix.head()

# + {"_uuid": "315bc6107a93f3926a64fd09ea9244e9281ee41f", "cell_type": "markdown"}
# ## Test set
# To use time tricks append test pairs to the matrix.
# -

test.head()

# + {"_uuid": "29d02bdb4fa768577607bf735b918ca81da85d41"}
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

# + {"_uuid": "177fbbab94c8057d67d61357d29581248468a74d"}
ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
time.time() - ts
# -

matrix.head()

# + {"_uuid": "233e394a6cebf36ef002dc76fef8d430026a52b3", "cell_type": "markdown"}
# ## Shops/Items/Cats features

# + {"_uuid": "7dfd5df3e2bcaee4c312f3979736f52c40f2560f"}
ts = time.time()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts
# -

matrix.head()

# + {"_uuid": "8358b291fdc8e0e7d1b5700974803b3f104715f7", "cell_type": "markdown"}
# ## Traget lags

# + {"_uuid": "9cd7bcc7643ce4545475e8e6f80d09a979aac42d"}
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

# + {"_uuid": "78bf7ece93ebc4629ad0e48cd6a9927788d8706d"}
ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts
# -

matrix.head()

# + {"_uuid": "c67bf4dbcef884ffe9d19c65d37bc4de1f287ef6", "cell_type": "markdown"}
# ## Mean encoded features

# + {"_uuid": "763aca242154ea10fa0a62fffadb4ef90e9532d6"}
ts = time.time()
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "fc9166c4e678ebb99d03566f1751b7d4b5c690d2"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "73f2552c403c5f67bbf07f28d69efcc015d00f32"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "c3948a9b206bc480b31385c29a713aa49747de19"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "bf98335755692f0d7666eeac2db1961692f09a16"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "3959603ea684eb3cbfd17d557399caa6e9da88e4"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "39f66d2e30f691237aa5d41ff9fc3a0eb7e9a788"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "87d57d01beb0830138dabae79b4022d4c6a9cc12"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "db1f0170ec4a6fd9894bc53b36f3166d4b26abcf"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "3cd5232ad63357dacebe9d223cc93dd669132bb7"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

# + {"_uuid": "00394f3694ae9c7093176eadac7abeaa79ff5467"}
ts = time.time()
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
# -

group.head()

matrix.head()

# + {"_uuid": "6bcea31d93ab035ca3fa1ed7c0afddbf602c414a", "cell_type": "markdown"}
# ## Trend features

# + {"_uuid": "0504e9613087237c255914d9ebd165fac4e88cd0", "cell_type": "markdown"}
# Price trend for the last six months.

# + {"_uuid": "0da2ded8502e273137991fd2bebbadaf19c19622"}
ts = time.time()
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts
# -

matrix.head()

# + {"_uuid": "17765ddb48f52abd88847a42c0a3ffe974e5b121", "cell_type": "markdown"}
# Last month shop revenue trend

# + {"_uuid": "e633be47f1a22b41487866ce67fb874bd296339e"}
ts = time.time()
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts

# + {"_uuid": "47e06af411b7d26cd93dad3d6735e48e5fbdee50", "cell_type": "markdown"}
# ## Special features

# + {"_uuid": "bb521e1f33d4124a3b90b47447bdb29150770b6e"}
matrix['month'] = matrix['date_block_num'] % 12

# + {"_uuid": "b4dc4d2ff86483989c4b74fc02a0d01ca68a5c75", "cell_type": "markdown"}
# Number of days in a month. There are no leap years.

# + {"_uuid": "e23f0201056b73368e3b70d4c36c6bb9e4a55291"}
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
# -

matrix.head()

# + {"_uuid": "7c096e86eb0043c0f6eeb899de24e28ca4c4e044", "cell_type": "markdown"}
# Months since the last sale for each shop/item pair and for item only. I use programing approach.
#
# <i>Create HashTable with key equals to {shop_id,item_id} and value equals to date_block_num. Iterate data from the top. Foreach row if {row.shop_id,row.item_id} is not present in the table, then add it to the table and set its value to row.date_block_num. if HashTable contains key, then calculate the difference beteween cached value and row.date_block_num.</i>

# + {"_uuid": "3458a7056c963167760921417d1f863f074f2b39"}
ts = time.time()
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         
time.time() - ts

# + {"_uuid": "28b29fae3906d870b4dc3064a7f359b6d3abf623"}
ts = time.time()
cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num         
time.time() - ts
# -

matrix.head()

# + {"_uuid": "61987e6adc1bec2ea897eec837c0253f7f73fdb5", "cell_type": "markdown"}
# Months since the first sale for each shop/item pair and for item only.

# + {"_uuid": "ad0869709bbada35726d5ca41dd913d817249f8e"}
ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts
# -

matrix.head()

# + {"_uuid": "966cb34ccfe849fbb3707d93270691cb8eef7a89", "cell_type": "markdown"}
# ## Final preparations
# Because of the using 12 as lag value drop first 12 months. Also drop all the columns with this month calculated values (other words which can not be calcucated for the test set).

# + {"_uuid": "04df1bc4240f409a5d4521c6f70c2ced44f7c3d4"}
ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts

# + {"_uuid": "48a14784050901f878b40f093e4bc34e07ecce05", "cell_type": "markdown"}
# Producing lags brings a lot of nulls.

# + {"_uuid": "8e5d8cb5cea9be28af4a0486cc1bf797e5b5c7ee"}
ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
time.time() - ts

# + {"_uuid": "00bf3fffc1b143d0555d03b9d79b5fd00d9d0dc9"}
matrix.columns

# + {"_uuid": "5f4e4c5c552daf8d4da6999ae4b63f13459b2887"}
matrix.info()

# + {"_uuid": "5d9988f8da8876f74092fbf827ceb6c61dd09d5e"}
matrix.to_pickle('data.pkl')
del matrix
del cache
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect();

# + {"_uuid": "b69932efb440af8f6435f3cd802fbcd15682af71", "cell_type": "markdown"}
# # Part 2, xgboost

# + {"_uuid": "a54364495b1818e9f069efa0c53500bf9e21d5f9"}
data = pd.read_pickle('data.pkl')

# + {"_uuid": "e5742775554b9e48e4d5c19784184069ad3eb9fb", "cell_type": "markdown"}
# Select perfect features

# + {"_uuid": "bfc928a916bb8b285b2fe90fb1a311cf2fbbf2e3"}
data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    #'date_shop_type_avg_item_cnt_lag_1',
    #'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    #'date_type_avg_item_cnt_lag_1',
    #'date_subtype_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]

# + {"_uuid": "11eb4f2f5ada18aa8993ec55e8c63e80758fc19e", "cell_type": "markdown"}
# Validation strategy is 34 month for the test set, 33 month for the validation set and 13-33 months for the train.

# + {"_uuid": "9af76d7b80064573a453e5e10c35b76fc31c47a4"}
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# + {"_uuid": "6ea5437e8949db6d3e54e68b7b0c18cd0befe38c"}
del data
gc.collect();

# + {"_uuid": "acef75c36501f808d45f81fc69f9708fc3283bc3"}
ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts

# + {"_uuid": "8ff5a80a22d046c5ca1cb27e938c757b607551d2"}
Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('xgb_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))

# + {"_uuid": "c8adc7c93323eb77baeceb2e8db17390b5c4deb3"}
plot_features(model, (10,14))
# -


