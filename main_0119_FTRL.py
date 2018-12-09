# -*- coding: utf-8 -*-
# @Time    : 1/15/18 9:03 AM
# @Author  : LeeYun
# @File    : main_0111_gru.py
'''Description :

'''
import os,numba,string,pickle,re,nltk,time,time_utils,logging_utils,gc
from vanila_conv1d_utils import vanila_conv1d_Regressor
from vanila_GRU_utils import vanila_GRU_Regressor
from vanila_FTRL_utils import vanila_FM_FTRL_Regressor,vanila_FTRL_Regressor
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
import wordbatch
from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL

THREAD=4

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

# brand_dict, name_dict, regex=_load('Data/brand_dict_name_dict_regex')

# # regularize name
# name_list=(
#     (['lebron soldier'], 'lebron_soldier'),
#     (['air force','air forces'], 'air_force'),
#     (['air max'], 'air_max'),
#     (['vs pink'], 'vs_pink'),
#     (['victoria secret','victoria secrets','v . s .'], 'victoria_\'_s_secret'),
#     (['f 21'], 'forever_21'),
#     (['lulu lemon','lulu'], 'lululemon'),
#     (['a . e .'], 'american_eagle'),
# )
# # map from keyword to brand
# brand_list=(
#     (['lebron_soldier','kyrie','air_force','air_max'],'nike'),
#     (['vs_pink'],'pink'),
#     (['vsx'],'victoria \' s secret'),
#     (['llr'],'lularoe'),
#     (['ipad','ipod','macbook','iphone'], 'apple'),
#     (['wii', 'gameboy', 'zelda', 'gamecube',],'nintendo'),
#     (['ae','aeo','aerie'],'american eagle'),
#     (['disneyland', 'minnie', 'mickey', 'woody', 'tsum'],'disney'),
# )
# name_dict,brand_dict={},{}
# for keywords,value in name_list:
#     for item in keywords:
#         name_dict[item]=value
# for keywords,brand in brand_list:
#     for item in keywords:
#         brand_dict[item]=brand+'2'
# del name_list,brand_list
#
# # Create a regular expression  from the dictionary keys
# @lru_cache(1024)
# def brand_check(brand: str,name: str):
#     if brand!=MISSVALUE: return brand
#     # check brand
#     if name!=MISSVALUE:
#         # regularize name
#         name = regex.sub(lambda mo: name_dict[mo.string[mo.start():mo.end()]], name)
#         words=name.split(" ")
#         for word in words:
#             if word in brand_dict:
#                 return brand_dict[word]
#     return brand
#
# def brand_fill(df: pd.DataFrame):
#     # fill brand
#     global regex
#     regex = re.compile("(%s)" % "|".join(map(re.escape, name_dict.keys())))
#     # _save('Data/brand_dict_name_dict_regex', [brand_dict, name_dict, regex])
#     return df.apply(lambda x: brand_check(x.values[0],x.values[1]),axis=1)

@lru_cache(1024)
def split_cat(text):
    text=text.split("/")
    if len(text)>=2:  return text[0],text[1]
    else: return text[0],MISSVALUE

@lru_cache(1024)
def len_splt(str,pat=' '):
    return len(str.split(pat))

def text_processor1(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'([a-z]+|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4'). \
        str.replace('\s+', ' '). \
        str.strip()


def text_processor2(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'[^\.!?@#&%$/\\0-9a-z]', ' '). \
        str.replace(r'(\.|!|\?|@|#|&|%|\$|/|\\|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4'). \
        str.replace('\s+', ' '). \
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
        columns = ['name', 'category_name', 'brand_name', 'item_description']
        for col in columns: merge[col].fillna(value=MISSVALUE, inplace=True)

        start_time = time.time()

        columns = ['item_description', 'name']
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        for col in columns:
            print(col)
            slices = [merge[col][:len1], merge[col][len1:len2], merge[col][len2:len3], merge[col][len3:]]
            dfvalue = []
            dfs = p.imap(text_processor1, slices)
            for df in dfs: dfvalue.append(df.values)
            merge[col] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
        p.close();
        slices, dfvalue, dfs, df, p = None, None, None, None, None;
        gc.collect()

        print('[{}] clean item_description completed'.format(time.time() - start_time))

        columns = ['brand_name', 'category_name']
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        for col in columns:
            slices = [merge[col][:len1], merge[col][len1:len2], merge[col][len2:len3], merge[col][len3:]]
            dfvalue = []
            dfs = p.imap(text_processor2, slices)
            for df in dfs: dfvalue.append(df.values)
            merge[col] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
        p.close();
        slices, dfvalue, dfs, df, p = None, None, None, None, None;
        gc.collect()

        merge['category_1'], merge['category_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
        return merge

    # def inductive_brand(merge: pd.DataFrame):
    #     # inductive the missing brand from name
    #     print((merge['brand_name'] == MISSVALUE).sum())
    #
    #     # supplement name_dict and brand_dict
    #     brand_frec = merge['brand_name'].value_counts()[1:200] # 150, 200, 250
    #     for i in range(brand_frec.shape[0]):
    #         text = brand_frec.index.values[i]
    #         text2 = '_'.join(text.split(' '))
    #         if text != text2: name_dict[text]=text2
    #         brand_dict[text2]=text+'2'
    #
    #     common_word=['pink','express','guess','beats','supreme']
    #     for item in common_word:
    #         if brand_dict[item]: brand_dict.pop(item)
    #
    #     p = multiprocessing.Pool(THREAD)
    #     length = merge.shape[0]
    #     len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
    #     slices = [merge[:len1], merge[len1:len2], merge[len2:len3], merge[len3:]]
    #     dfvalue = []
    #     dfs = p.imap(brand_fill, slices)
    #     for df in dfs: dfvalue.append(df)
    #     brand_name = pd.concat((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]), ignore_index=True)
    #     print((brand_name == MISSVALUE).sum())
    #     p.close(); slices,dfvalue,dfs,df,p=None,None,None,None,None; gc.collect()
    #     return brand_name

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
        # dfAll=dfAll[:100000]
        dfAll = textclean(dfAll)
        # dfAll['brand_name'] = inductive_brand(dfAll[['brand_name','name']])
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
    y_train = ((np.log1p(dfAll.price) - PRICE_MEAN) / PRICE_STD).values[:TRAIN_SIZE]
    dfAll.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)
    return dfAll, submission, y_train

def Label_Encoder(df):
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
    def __init__(self, sparse_merge,price):
        self.sparse_merge=sparse_merge
        self.price=price
        self.total_len=TRAIN_SIZE
        self.splitter=self.split_data()
        self.feature_dim=sparse_merge.shape[1]

    def split_data(self):
        splitter=[]
        cv = KFold(n_splits=cv_fold, shuffle=False, random_state=2018)
        for train_ids, valid_ids in cv.split(self.price):
            splitter.append([train_ids.copy(), valid_ids.copy()])
        return splitter

    def _get_train_valid_data(self,i):
        train_ind,test_ind=self.splitter[i]
        X_train=self.sparse_merge[train_ind]
        X_valid=self.sparse_merge[test_ind]
        y_train=self.price[train_ind]
        y_valid=self.price[test_ind]
        return X_train, y_train, X_valid, y_valid, test_ind

class Learner:
    def __init__(self, learner_name, param_dict,feature_dim):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner(feature_dim)

    def __str__(self):
        return self.learner_name

    def _get_learner(self,feature_dim):
        if self.learner_name in ["FM_FTRL",'best_FM_FTRL']:
            return vanila_FM_FTRL_Regressor(param_dict=self.param_dict,feature_dim=feature_dim)
        if self.learner_name in ["FTRL",'best_FTRL']:
            return vanila_FTRL_Regressor(param_dict=self.param_dict,feature_dim=feature_dim)

    def fit(self, X_train, y_train,X_valid, y_valid):
        self.learner.fit(X_train, y_train,X_valid, y_valid)
        return self

    def predict(self, X_valid):
        y_pred = self.learner.predict(X_valid)
        return y_pred

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
            X_train, y_train, X_valid, y_valid, test_ind= self.feature._get_train_valid_data(i)
            # fit
            self.learner.fit(X_train, y_train,X_valid, y_valid)
            y_pred = self.learner.predict(X_valid)
            
            #_save('Data/Tfm', [y_pred,y_valid])
            # analyze loss
            # _save('Data/loss', [X_valid,y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD])

            # fit
            rmse_cv[i] = self.compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
            print('loss: ',rmse_cv[i])
            # log
            self.logger.info("      {:>3}    {:>8}".format(i+1, np.round(rmse_cv[i],6)))

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
    def __init__(self, learner_name, sparse_merge,price,logger):
        self.learner_name = learner_name
        self.logger = logger
        self.feature = self._get_feature(sparse_merge[:TRAIN_SIZE],price[:TRAIN_SIZE])
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self,sparse_merge,price):
        feature = Feature(sparse_merge,price)
        return feature

    def _obj(self, param_dict):
        self.trial_counter += 1
        print('learner_name: %s, trial_counter: %s'%(self.learner_name,self.trial_counter))
        start_time = time.time()
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict,self.feature.feature_dim)
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
    feature_vectorized_file_name='Data/feature_vectorized2'
    if os.path.exists(feature_vectorized_file_name)==False:
        sparse_merge,price=_load(feature_vectorized_file_name)
        print(sparse_merge.shape)
    else:
        ########################################################################
        start_time = time.time()
        merge, submission, price = get_extract_feature()
        merge=merge[:TRAIN_SIZE]

        merge['item_condition_id'] = merge['item_condition_id'].astype('category')
        print('[{}] Convert categorical completed'.format(time.time() - start_time))

        # vectorize features
        wb = CountVectorizer()
        X_category2 = wb.fit_transform(merge['category_2'])
        X_category3 = wb.fit_transform(merge['category_name'])
        X_brand2 = wb.fit_transform(merge['brand_name'])
        print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

        lb = LabelBinarizer(sparse_output=True)
        X_brand = lb.fit_transform(merge['brand_name'])
        X_category1 = lb.fit_transform(merge['category_1'])
        X_category4 = lb.fit_transform(merge['category_name'])
        print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

        X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

        # hand feature
        for col in merge.columns:
            if ('Len' in col) or ('Frec' in col):
                merge[col] = np.log1p(merge[col])
                merge[col] = merge[col] / merge[col].max()

        hand_feature = ['brand_name_Frec', 'item_description_wordLen', 'brand_name_name_Intsct',
                        'brand_name_item_description_Intsct']
        X_hand_feature = merge[hand_feature].values

        name_w1=param_space_best_WordBatch['name_w1']
        name_w2=param_space_best_WordBatch['name_w2']
        desc_w1=param_space_best_WordBatch['desc_w1']
        desc_w2=param_space_best_WordBatch['desc_w2']

        wb = wordbatch.WordBatch(normalize_text=None,extractor=(WordBag, {
            "hash_ngrams": 2,
            "hash_ngrams_weights": [name_w1, name_w2],
            "hash_size": 2 ** 28,
            "norm": None,
            "tf": 'binary',
            "idf": None,
        }), procs=8)
        wb.dictionary_freeze = True
        X_name = wb.fit_transform(merge['name'])
        del (wb)
        X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 2, 0, 1), dtype=bool)]
        print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

        merge['item_description'] = merge['category_2'].map(str)+' .#d3 .#d3 '+\
                                    merge['name'].map(str)+' .#d3 .#d3 '+\
                                    merge['item_description'].map(str)
        
        wb = wordbatch.WordBatch(normalize_text=None,extractor=(WordBag, {
            "hash_ngrams": 3,
            "hash_ngrams_weights": [desc_w1, desc_w2,0.7],
            "hash_size": 2 ** 28,
            "norm": "l2",
            "tf": 1.0,
            "idf": None
        }), procs=8)
        wb.dictionary_freeze = True
        X_description = wb.fit_transform(merge['item_description'])
        del (wb)
        X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 6, 0, 1), dtype=bool)]
        print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

        sparse_merge = hstack((X_dummies, X_brand,X_brand2, X_category1, X_category2, X_category3, X_category4, X_hand_feature,X_name,X_description)).tocsr()

        print(X_dummies.shape, X_brand.shape, X_brand2.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_category4.shape,
              X_hand_feature.shape,X_name.shape,X_description.shape,sparse_merge.shape)

        _save(feature_vectorized_file_name, [sparse_merge,price])
        print('[{}] data saved.'.format(time.time() - start_time))

    ########################################################################
    # use hyperopt to find the best parameters of the model
    # use 3 fold cross validation

    # learner_name='best_FTRL'
    # learner_name='FTRL'
    learner_name='best_FM_FTRL'
    #learner_name='FM_FTRL'
    print(learner_name)
    logname = "[Learner@%s]_hyperopt_%s.log"%(learner_name,time_utils._timestamp())
    logger = logging_utils._get_logger('Log', logname)
    logger.info('start')

    optimizer=TaskOptimizer(learner_name,sparse_merge,price,logger)
    optimizer.run()

    a=12

if __name__ == "__main__":
    main()

























































