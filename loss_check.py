# -*- coding: utf-8 -*-
# @Time    : 2/7/18 5:08 PM
# @Author  : LeeYun
# @File    : loss_check.py
'''Description :

'''
import pickle,pandas as pd,numpy as np
from sklearn.model_selection import KFold
def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

X_valid,y_valid, y_pred=_load('Data/loss')

length=y_pred.shape[0]
def read_file(name: str):
    source = '../input/%s.tsv' % name
    df = pd.read_table(source, engine='c')
    return df
dftrain = read_file('train')
dftrain = dftrain[dftrain.price != 0]
dftrain=dftrain[:length]

PRICE_MEAN = 2.98081628517
dftrain['pred']=np.expm1(y_pred +PRICE_MEAN)
dftrain['valid']=np.expm1(y_valid +PRICE_MEAN)
data=dftrain.iloc[np.argsort(-np.abs(y_pred-y_valid))]
data=data[['name', 'category_name', 'brand_name', 'item_description', 'price','pred']]

a=1
# dftrain[(dftrain.brand_name.isnull().astype(np.int8) + (dftrain.price>1000).astype(np.int8))==2]

# PRICE_MEAN = 2.98081628517
# price = (np.log1p(dftrain.price) - PRICE_MEAN)
#
# a=price[:length].values
#
#
# def split_data():
#     splitter = []
#     cv = KFold(n_splits=5, shuffle=False, random_state=2018)
#     for train_ids, valid_ids in cv.split(price):
#         splitter.append([train_ids.copy(), valid_ids.copy()])
#     return splitter
#
# splitter=split_data()
# splitter[0]


a=1






















































