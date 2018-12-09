# -*- coding: utf-8 -*-
# @Time    : 1/15/18 9:03 AM
# @Author  : LeeYun
# @File    : main_0111_gru.py
'''Description :

'''
import os,numba,string,pickle,re,nltk,time,time_utils,logging_utils,gc
from vanila_conv1d_utils import vanila_conv1d_Regressor
from vanila_GRU_utils import vanila_GRU_Regressor
import pandas as pd
import numpy as np
from config import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from fastcache import clru_cache as lru_cache
from nltk.stem.porter import PorterStemmer
import multiprocessing
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix, hstack
from scipy.stats import pearsonr
from sklearn.feature_extraction import stop_words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

THREAD=4

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

@lru_cache(1024)
def split_cat(text: str):
    text = text.split("/")
    if len(text) >= 2:
        return text[0], text[1]
    else:
        return text[0], MISSVALUE

@lru_cache(1024)
def len_splt(str: str):
    return len(str.split(' '))

def text_processor(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'[^\.!?@#&%$/\\0-9a-z]', ' '). \
        str.replace(r'(\.|!|\?|@|#|&|%|\$|/|\\|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4').\
        str.replace('\s+', ' ').\
        str.strip()

def intersect_cnt(dfs):
    intersect=np.empty(dfs.shape[0])
    for i in range(dfs.shape[0]):
        obs_tokens=set(dfs[i,0].split(" "))
        target_tokens = set(dfs[i,1].split(" "))
        intersect[i]=len(obs_tokens.intersection(target_tokens))/(obs_tokens.__len__()+1)
    return intersect


def get_extract_feature():
    def read_file(name: str):
        source = '../input/%s.tsv' % name
        df = pd.read_table(source, engine='c')
        return df

    def textclean(merge: pd.DataFrame):
        columns = ['name', 'category_name', 'item_description'] #'brand_name', 
        for col in columns: merge[col].fillna(value=MISSVALUE, inplace=True)
        merge['brand_name'].fillna(value='', inplace=True)

        columns = ['item_description', 'name', 'brand_name', 'category_name']
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        for col in columns:
            slices = [merge[col][:len1], merge[col][len1:len2], merge[col][len2:len3], merge[col][len3:]]
            dfvalue = []
            dfs = p.imap(text_processor, slices)
            for df in dfs: dfvalue.append(df.values)
            merge[col] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
           
            
        merge['category_1'], merge['category_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
        p.close();
        slices, dfvalue, dfs, df, p = None, None, None, None, None;
        gc.collect()
        return merge

    def get_cleaned_data():
        dftrain = read_file('train')
        dftest = read_file('test')

        # # Make a stage 2 test by copying test five times...
        # test1 = dftest.copy()
        # test2 = dftest.copy()
        # test3 = dftest.copy()
        # test4 = dftest.copy()
        # test5 = dftest.copy()
        # dftest = pd.concat([test1, test2, test3, test4, test5], axis=0)
        # test1 = None
        # test2 = None
        # test3 = None
        # test4 = None
        # test5 = None

        dftrain = dftrain[dftrain.price != 0]
        dfAll = pd.concat((dftrain, dftest), ignore_index=True)
        dfAll = textclean(dfAll)
        submission: pd.DataFrame = dftest[['test_id']]
        dftrain, dftest = None, None;
        gc.collect()
        return dfAll, submission

    def add_Frec_feat(dfAll: pd.DataFrame, col: str):
        s = dfAll[col].value_counts()
        s[MISSVALUE] = 0
        dfAll = dfAll.merge(s.to_frame(name=col + '_Frec'), left_on=col, right_index=True, how='left')
        s = None
        return dfAll

    dfAll, submission = get_cleaned_data()
    print('data cleaned')

    # add the Frec features
    columns = ['brand_name']
    print('add the Frec features')
    for col in columns: dfAll = add_Frec_feat(dfAll, col)

    # intersection count between 'item_description','name','brand_name','category_name'
    columns = [['brand_name', 'name'], ['brand_name', 'item_description']]
    p = multiprocessing.Pool(THREAD)
    length = dfAll.shape[0]
    len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
    for col in columns:
        slices = [dfAll[col].values[:len1], dfAll[col].values[len1:len2], dfAll[col].values[len2:len3],
                  dfAll[col].values[len3:]]
        dfvalue = []
        dfs = p.imap(intersect_cnt, slices)
        for df in dfs: dfvalue.append(df)
        dfAll['%s_%s_Intsct' % (col[0], col[1])] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
    p.close();
    slices, dfvalue, dfs, df, p = None, None, None, None, None;
    gc.collect()

    # count item_description length
    dfAll['item_description_wordLen'] = dfAll.item_description.apply(lambda x: len_splt(x))
    # nomalize price
    y_train = ((np.log1p(dfAll.price) - PRICE_MEAN) / PRICE_STD).values[:TRAIN_SIZE].reshape(-1, 1).astype(np.float32)
    dfAll['norm_price'] = (np.log1p(dfAll.price)-PRICE_MEAN)/PRICE_STD
    dfAll.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)
    return dfAll, submission, y_train

def Label_Encoder(df: pd.Series):
    le = LabelEncoder()
    return le.fit_transform(df)

def Item_Tokenizer(df: pd.Series):
    # do not concern the words only appeared in dftest
    tok_raw = Tokenizer(num_words=NumWords, filters='')
    tok_raw.fit_on_texts(df[:TRAIN_SIZE])
    return tok_raw.texts_to_sequences(df), min(tok_raw.word_counts.__len__() + 1, NumWords)

def _corr(x, y_train):
    corr = pearsonr(x.flatten(), y_train)[0]
    if str(corr) == "nan":
        corr = 0.
    return corr

class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            "see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict

class Feature:
    def __init__(self, data,Item_size,hand_feature):
        self.data=data
        self.Item_size=Item_size
        self.Item_size['hand_feature']=hand_feature
        self.hand_feature=hand_feature
        self.total_len=TRAIN_SIZE
        self.splitter=self.split_data()

    def split_data(self):
        splitter=[]
        cv = KFold(n_splits=cv_fold, shuffle=False, random_state=2018)
        for train_ids, valid_ids in cv.split(self.data):
            splitter.append([train_ids.copy(), valid_ids.copy(), train_ids.shape[0], valid_ids.shape[0]])
        return splitter

    def preprocess_data(self,param_dict):
        data=self.data
        fname='Data/seq_name_desc.pkl'
        develop=0
        if develop:
            self.X_seq_item_description= pad_sequences(data['Seq_item_description'], maxlen=param_dict['description_Len'])
            self.X_seq_name= pad_sequences(data['Seq_name'], maxlen=param_dict['name_Len'])
            _save(fname, [self.X_seq_item_description,self.X_seq_name])
        else:
            self.X_seq_item_description, self.X_seq_name=_load(fname)
        self.X_brand_name=data.brand_name.values.reshape(-1,1)
        self.X_category_1=data.category_1.values.reshape(-1,1)
        self.X_category_2=data.category_2.values.reshape(-1,1)
        self.X_category_name=data.category_name.values.reshape(-1,1)
        self.X_item_condition_id=(data.item_condition_id.values.astype(np.int32)-1).reshape(-1,1)
        self.X_shipping=(data.shipping.values-data.shipping.values.mean()/data.shipping.values.std()).reshape(-1,1)
        self.X_hand_feature=(data[self.hand_feature].values-data[self.hand_feature].values.mean(axis=0))/data[self.hand_feature].values.std(axis=0)
        self.Y_price=data.norm_price.values.reshape(-1,1)

    def _get_train_valid_data(self,i):
        train_ind,test_ind,train_len,valid_len=self.splitter[i]
        X_train=dict(
            X_seq_item_description = self.X_seq_item_description[train_ind],
            X_seq_name = self.X_seq_name[train_ind],
            X_brand_name = self.X_brand_name[train_ind],
            X_category_1 = self.X_category_1[train_ind],
            X_category_2 = self.X_category_2[train_ind],
            X_category_name = self.X_category_name[train_ind],
            X_item_condition_id = self.X_item_condition_id[train_ind],
            X_shipping = self.X_shipping[train_ind],
            X_hand_feature = self.X_hand_feature[train_ind],
        )
        X_valid=dict(
            X_seq_item_description = self.X_seq_item_description[test_ind],
            X_seq_name = self.X_seq_name[test_ind],
            X_brand_name = self.X_brand_name[test_ind],
            X_category_1 = self.X_category_1[test_ind],
            X_category_2 = self.X_category_2[test_ind],
            X_category_name = self.X_category_name[test_ind],
            X_item_condition_id = self.X_item_condition_id[test_ind],
            X_shipping = self.X_shipping[test_ind],
            X_hand_feature = self.X_hand_feature[test_ind],
        )
        y_train=self.Y_price[train_ind]
        y_valid=self.Y_price[test_ind]
        return X_train, y_train, X_valid, y_valid,train_len,valid_len

class Learner:
    def __init__(self, learner_name, param_dict,Item_size):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner(Item_size)

    def __str__(self):
        return self.learner_name

    def _get_learner(self,Item_size):
        if self.learner_name in ["vanila_con1d",'best_vanila_con1d']:
            return vanila_conv1d_Regressor(param_dict=self.param_dict,Item_size=Item_size)
        if self.learner_name in ["vanila_GRU",'best_vanila_GRU']:
            return vanila_GRU_Regressor(param_dict=self.param_dict,Item_size=Item_size)
        if self.learner_name in ["con1d_ensemble",'best_con1d_ensemble']:
            from config import param_space_best_vanila_con1d
            for key,values in param_space_best_vanila_con1d.items():
                self.param_dict[key]=values
            return vanila_conv1d_Regressor(param_dict=self.param_dict,Item_size=Item_size)
        if self.learner_name in ["GRU_ensemble",'best_GRU_ensemble']:
            from config import param_space_best_vanila_GRU
            for key,values in param_space_best_vanila_GRU.items():
                self.param_dict[key]=values
            return vanila_GRU_Regressor(param_dict=self.param_dict,Item_size=Item_size)

    def fit(self, X_train, y_train, X_valid, y_valid,train_len,valid_len):
        pred=self.learner.fit(X_train, y_train, X_valid, y_valid,train_len,valid_len)
        return pred

    def predict(self, X_valid,valid_len):
        y_pred = self.learner.predict(X_valid,valid_len)
        return y_pred

    def close_session(self):
        self.learner.close_session()

class Task:
    def __init__(self, learner, feature, suffix, logger):
        self.learner = learner
        self.suffix = suffix
        self.logger = logger
        self.feature = feature
        self.rmse_cv_mean = 0
        self.rmse_cv_std = 0
        self.verbose = 1

    def __str__(self):
        return "[Learner@%s]%s"%(str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix,k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix,k,v))

    def compute_loss(self,pred, y):
        return np.sqrt(np.mean(np.square(pred - y)))

    def cv(self):
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("      %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(self.learner.param_dict)
            self.logger.info("Result")
            self.logger.info("      Run      RMSE")

        cv_fold2=1
        rmse_cv = np.zeros((cv_fold2))
        for i in range(cv_fold2):
            X_train, y_train, X_valid, y_valid,train_len,valid_len = self.feature._get_train_valid_data(i)

            # # fit
            # self.learner.fit(X_train, y_train, X_valid, y_valid,train_len,valid_len)
            # y_pred = self.learner.predict(X_valid,valid_len)
            # rmse_cv[i] = self.compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
            # print('loss: ',rmse_cv[i])

            # fit
            y_pred = self.learner.fit(X_train, y_train, X_valid, y_valid,train_len,valid_len)
            rmse_cv[i] = self.compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
            print('loss: ',rmse_cv[i])

            # log
            self.logger.info("      {:>3}    {:>8}".format(i+1, np.round(rmse_cv[i],6)))

            # if rmse_cv[i]>threshold_dict[self.learner.learner_name]:
            #     for j in range(i+1,cv_fold):
            #         rmse_cv[j] = rmse_cv[i]
            #         self.logger.info("      {:>3}    {:>8}".format(i + 1, np.round(rmse_cv[i], 6)))
            #     break
        self.learner.close_session()

        self.rmse_cv_mean = np.mean(rmse_cv)
        self.rmse_cv_std = np.std(rmse_cv)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("RMSE")
            self.logger.info("      Mean: %.6f"%self.rmse_cv_mean)
            self.logger.info("      Std: %.6f"%self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("      %d mins"%_min)
            else:
                self.logger.info("      %d secs"%_sec)
            self.logger.info("-"*50)
        return self
    def go(self):
        self.cv()
        return self

class TaskOptimizer:
    def __init__(self, learner_name, merge, Item_size, hand_feature,logger):
        self.learner_name = learner_name
        self.logger = logger
        self.feature = self._get_feature(merge[:TRAIN_SIZE].copy(),Item_size,hand_feature)
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self,data,Item_size,hand_feature):
        feature = Feature(data,Item_size,hand_feature)
        return feature

    def _obj(self, param_dict):
        self.trial_counter += 1
        print('learner_name: %s, trial_counter: %s'%(self.learner_name,self.trial_counter))
        start_time = time.time()
        param_dict = self.model_param_space._convert_int_param(param_dict)
        self.feature.preprocess_data(param_dict)
        learner = Learner(self.learner_name, param_dict,self.feature.Item_size)
        suffix = "_[Id@%s]"%str(self.trial_counter)
        self.task = Task(learner, self.feature, suffix, self.logger)
        self.task.go()
        print('time: ', time.time()-start_time)
        ret = {
            "loss": self.task.rmse_cv_mean,
            "attachments": {
                "std": self.task.rmse_cv_std,
            },
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        start = time.time()
        trials = Trials()
        best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, hp_iter, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        trial_rmses = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_rmses)
        best_rmse_mean = trial_rmses[best_ind]
        best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-"*50)
        self.logger.info("Best RMSE")
        self.logger.info("      Mean: %.6f"%best_rmse_mean)
        self.logger.info("      std: %.6f"%best_rmse_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)


def main():
    ########################################################################
    file_name='Data/feature_processed'
    if os.path.exists(file_name)==False:
        merge, Item_size, hand_feature = _load('Data/feature_processed')
        print(hand_feature)
    else:
        start_time = time.time()
        merge, submission, y = get_extract_feature()

        merge['item_condition_id'] = merge['item_condition_id'].astype('category')
        print('[{}] Convert categorical completed'.format(time.time() - start_time))
        
        Item_size={}
        # Label_Encoder brand_name + category
        columns=['category_1','category_2','category_name','brand_name']
        p = multiprocessing.Pool(4)
        dfs = p.imap(Label_Encoder, [merge[col] for col in columns])
        for col, df in zip(columns, dfs):
            merge[col] = df
            Item_size[col]=merge[col].max()+1
        print('[{}] Label Encode `brand_name` and `categories` completed.'.format(time.time() - start_time))

        # sequance item_description,name
        columns=['item_description','name']
        p = multiprocessing.Pool(4)
        dfs = p.imap(Item_Tokenizer, [merge[col] for col in columns])
        for col, df in zip(columns, dfs):
            merge['Seq_'+col],Item_size[col] = df
        print('[{}] sequance `item_description` and `name` completed.'.format(time.time() - start_time))
        print(Item_size)

        # hand feature
        columns = ['brand_name_Frec', 'item_description_wordLen']
        for col in columns:
            merge[col] = np.log1p(merge[col])
            merge[col] = merge[col] / merge[col].max()

        hand_feature = ['brand_name_Frec',
                        'item_description_wordLen',
                        'brand_name_name_Intsct',
                        'brand_name_item_description_Intsct']

        _save(file_name, [merge,Item_size,hand_feature])
        print('[{}] data saved.'.format(time.time() - start_time))

    ########################################################################
    # use hyperopt to find the best parameters of the model
    # use 3 fold cross validation

    #learner_name = 'best_GRU_ensemble'
    # learner_name = 'GRU_ensemble'
    #learner_name = 'best_con1d_ensemble'
    # learner_name = 'con1d_ensemble'
    learner_name='best_vanila_con1d'
    # learner_name='best_vanila_GRU'
    #learner_name='vanila_con1d'
    # learner_name='vanila_GRU'
    print(learner_name)
    logname = "[Learner@%s]_hyperopt_%s.log"%(learner_name,time_utils._timestamp())
    logger = logging_utils._get_logger('Log', logname)
    logger.info('start')

    optimizer=TaskOptimizer(learner_name,merge, Item_size, hand_feature,logger)
    optimizer.run()

if __name__ == "__main__":
    main()

























































