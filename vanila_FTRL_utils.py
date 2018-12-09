# -*- coding: utf-8 -*-
# @Time    : 1/18/18 9:08 AM
# @Author  : LeeYun
# @File    : vanila_GRU_utils2.py
'''Description :

'''
import time,wordbatch,gc,pickle
import numpy as np
from wordbatch.models import FTRL, FM_FTRL
from wordbatch.extractors import WordBag
from scipy.sparse import csr_matrix, hstack
from config import param_space_best_FM_FTRL
from config import PRICE_STD


def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def compute_loss(pred, y):
    return np.sqrt(np.mean(np.square(pred - y)))

class vanila_FM_FTRL_Regressor:
    def __init__(self, param_dict,feature_dim):
        alpha=param_dict['alpha']
        beta=param_dict['beta']
        L1=param_dict['L1']
        L2=param_dict['L2']
        alpha_fm=param_dict['alpha_fm']
        init_fm=param_dict['init_fm']
        D_fm=param_dict['D_fm']
        e_noise=param_dict['e_noise']
        self.iters=param_dict['iters']

        self.model = FM_FTRL(alpha=alpha, beta=beta, L1=L1, L2=L2, D=feature_dim, alpha_fm=alpha_fm, L2_fm=0.0,
                        init_fm=init_fm, D_fm=D_fm, e_noise=e_noise, iters=self.iters, inv_link="identity", threads=6)

    def fit(self, X_train, y_train,X_valid=0, y_valid=0):
#         for i in range(self.iters-1):
#             self.model.fit(X_train, y_train)
#             y_pred=self.model.predict(X_valid)
#             loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, y_pred * PRICE_STD)
#             print('iter: %d, loss: %f'%(i,loss))
        
#         slice_len=X_train.shape[0]//4
#         weight=0.7
#         y_pred=np.zeros(X_valid.shape[0])
#         for i in range(4):
#             start=i*slice_len
#             rear=(i+1)*slice_len if i!=3 else X_train.shape[0]
#             self.model.fit(X_train[start:rear], y_train[start:rear])
#             tmp=self.model.predict(X_valid)
#             loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, tmp * PRICE_STD)
#             print('iter: %d, loss: %f'%(i,loss))
#             y_pred=y_pred+tmp*weight
#             weight+=0.2
        
#         y_pred/=4
#         loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, y_pred * PRICE_STD)
#         print('loss: %f'%(loss))
        
        self.model.fit(X_train, y_train)
        #y_pred = self.model.predict(X_valid)
        #loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, y_pred * PRICE_STD)
        #print('loss: %f'%(loss))
        #self.model2.fit(X_train, y_train)
        #y_pred2 = self.model2.predict(X_valid)
        #loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, y_pred2 * PRICE_STD)
        #print('loss: %f'%(loss))
        #loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, (y_pred+y_pred2)/2 * PRICE_STD)
        #print('loss: %f'%(loss))
        #loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, (y_pred*0.6+y_pred2*0.4) * PRICE_STD)
        #print('loss: %f'%(loss))
        #loss=compute_loss(y_valid.reshape(-1) * PRICE_STD, (y_pred*0.7+y_pred2*0.3) * PRICE_STD)
        #print('loss: %f'%(loss))
        #_save('Data/FM_ensemble',[y_valid,y_pred,y_pred2])

    def predict(self,X_valid):
        return self.model.predict(X_valid)

class vanila_WorldBatch_Regressor:
    def __init__(self, param_dict):
        name_w1=param_dict['name_w1']
        name_w2=param_dict['name_w2']
        desc_w1=param_dict['desc_w1']
        desc_w2=param_dict['desc_w2']
        self.dict=param_space_best_FM_FTRL

        self.wb_name = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
            "hash_ngrams": 2,
            "hash_ngrams_weights": [name_w1, name_w2],
            "hash_size": 2 ** 28,
            "norm": None,
            "tf": 'binary',
            "idf": None,
        }), procs=8)
        self.wb_name.dictionary_freeze = True

        self.wb_desc = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
            "hash_ngrams": 3,
            "hash_ngrams_weights": [desc_w1, desc_w2, 0.7],
            "hash_size": 2 ** 28,
            "norm": "l2",
            "tf": 1.0,
            "idf": None
        }), procs=8)
        self.wb_desc.dictionary_freeze = True

    def predict(self,sparse_merge,desc,name,y_train,train_ind, test_ind):
        X_name = self.wb_name.fit_transform(name).astype(np.float32)
        X_name = X_name[:, np.array(np.clip(X_name[train_ind].getnnz(axis=0) - 2, 0, 1), dtype=bool)]
        print(X_name.shape)
        X_name = X_name[:, np.array(np.clip(X_name[test_ind].getnnz(axis=0), 0, 1), dtype=bool)]
        print(X_name.shape)
        print('Vectorize `name` completed.')

        X_description = self.wb_desc.fit_transform(desc).astype(np.float32)
        X_description = X_description[:, np.array(np.clip(X_description[train_ind].getnnz(axis=0) - 6, 0, 1), dtype=bool)]
        print(X_description.shape)
        X_description = X_description[:, np.array(np.clip(X_description[test_ind].getnnz(axis=0), 0, 1), dtype=bool)]
        print(X_description.shape)
        print('Vectorize `description` completed.')

        sparse_merge=hstack((sparse_merge,X_name,X_description)).tocsr()
        print(X_name.shape, X_description.shape, sparse_merge.shape)
        desc, name=None,None; gc.collect()

        X_train, X_test=sparse_merge[train_ind],sparse_merge[test_ind]
        feature_dim=sparse_merge.shape[1]

        model=vanila_FM_FTRL_Regressor(self.dict,feature_dim)
        model.fit(X_train, y_train)
        return model.predict(X_test)

class vanila_FTRL_Regressor:
    def __init__(self, param_dict,feature_dim):
        alpha=param_dict['alpha']
        beta=param_dict['beta']
        L1=param_dict['L1']
        L2=param_dict['L2']
        iters=param_dict['iters']

        self.model = FTRL(alpha=alpha, beta=beta, L1=L1, L2=L2, D=feature_dim, iters=iters, inv_link="identity",
                     threads=6)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self,X_valid):
        return self.model.predict(X_valid)










































