# -*- coding: utf-8 -*-
# @Time    : 1/20/18 9:29 PM
# @Author  : LeeYun
# @File    : final_submission.py
'''Description :
0.41260
'''
import os, string, pickle, re, time, gc, multiprocessing, wordbatch
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from fastcache import clru_cache as lru_cache
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from wordbatch.models import FM_FTRL
from wordbatch.extractors import WordBag

TRAIN_SIZE = 1481661
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 70000
THREAD = 4
MISSVALUE = 'missvalue'

# 0.41260
param_space_best_FM_FTRL = {
    'alpha': 0.023417717538033785,
    'beta': 0.0007479811328334067,
    'L1': 2.6904652243116978e-05,
    'L2': 0.011450866384106273,
    'alpha_fm': 0.012987693454921333,
    'init_fm': 0.03466079208592963,
    'D_fm': 253,
    'e_noise': 0.00045792625420732223,
    'iters': 11,
}

param_space_best_WordBatch={
    'desc_w1': 1.3740067995315037,
    'desc_w2': 1.0248685266832964,
    'name_w1': 2.1385527373939834,
    'name_w2': 0.3894761681383836,
}

# # regularize name
# name_list=(
#     (['lebron soldier'], 'lebron_soldier'),
#     (['air force','air forces'], 'air_force'),
#     (['air max'], 'air_max'),
#     (['vs pink'], 'vs_pink'),
#     (['victoria s secret','victoria secret','victoria secrets','v . s .'], 'victoria_secret'),
#     (['forever 21','f 21'], 'forever_21'),
#     (['game boy'], 'game_boy'),
#     (['lulu lemon','lulu'], 'lululemon'),
#     (['michael kors'], 'michael_kors'),
#     (['american eagle','ae','a . e .'], 'american_eagle'),
#     (['rae dunn'], 'rae_dunn'),
#     (['m . a . c'], 'm.a.c'),
#     (['bath & body works'], 'bath_&_body_works'),
#     (['under armour'], 'under_armour'),
#     (['old navy'], 'old_navy'),
#     (['carter s'], 'carter_s'),
#     (['the north face'], 'the_north_face'),
#     (['urban decay'], 'urban_decay'),
#     (['too faced'], 'too_faced'),
#     (['brandy melville'], 'brandy_melville'),
#     (['kate spade'], 'kate_spade'),
#     (['kendra scott'], 'kendra_scott'),
#     (['ugg australia'], 'ugg_australia'),
#     (['ralph lauren'], 'ralph_lauren'),
#     (['charlotte russe'], 'charlotte_russe'),
#     (['vera bradley'], 'vera_bradley'),
#     (['h & m'], 'h&m'),
#     (['tory burch'], 'tory_burch'),
#     (['air jordan'], 'air_jordan'),
# )
# # map from keyword to brand
# brand_list=(
#     (['nike','lebron_soldier','kyrie','air_force','air_max'],'nike'),
#     (['vs_pink'],'pink'),
#     (['victoria_secret','vsx'],'victoria s secret'),
#     (['lularoe','llr'],'lularoe'),
#     (['ipad','ipod','apple','macbook','iphone'], 'apple'),
#     (['forever_21'], 'forever 21'),
#     (['nintendo', 'wii', 'gameboy', 'zelda', 'gamecube',],'nintendo'),
#     (['lululemon'],'lululemon'),
#     (['michael_kors'],'michael kors'),
#     (['american_eagle','aeo','aerie'],'american eagle'),
#     (['rae_dunn'],'rae dunn'),
#     (['disney', 'disneyland', 'minnie', 'mickey', 'woody', 'tsum'],'disney'),
#     (['mlb'],'mlb'),
#     (['m.a.c'],'mac'),
#     (['sephora'],'sephora'),
#     (['bath_&_body_works'],'bath & body works'),
#     (['adidas'],'adidas'),
#     (['funko'],'funko'),
#     (['under_armour'],'under armour'),
#     (['sony'],'sony'),
#     (['old_navy'],'old navy'),
#     (['hollister'],'hollister'),
#     (['carter_s'],'carter s'),
#     (['the_north_face'],'the north face'),
#     (['urban_decay'],'urban decay'),
#     (['too_faced'],'too faced'),
#     (['xbox'], 'xbox'),
#     (['brandy_melville'], 'brandy melville'),
#     (['kate_spade'], 'kate spade'),
#     (['kendra_scott'], 'kendra scott'),
#     (['tarte'], 'tarte'),
#     (['ugg_australia'], 'ugg australia'),
#     (['vans'], 'vans'),
#     (['ralph_lauren'], 'ralph lauren'),
#     (['charlotte_russe'], 'charlotte russe'),
#     (['vera_bradley'], 'vera bradley'),
#     (['samsung'], 'samsung'),
#     (['senegence'], 'senegence'),
#     (['converse'], 'converse'),
#     (['h&m'], 'h&m'),
#     (['tory_burch'], 'tory burch'),
#     (['air_jordan'], 'air jordan'),
# )
# name_dict={}
# for keywords,value in name_list:
#     for item in keywords:
#         name_dict[item]=value#+'** '
# brand_dict={}
# for keywords,brand in brand_list:
#     for item in keywords:
#         brand_dict[item]=brand+'2'
# del name_list,brand_list
#
# regex = re.compile("(%s)" % "|".join(map(re.escape, name_dict.keys())))
# # Create a regular expression  from the dictionary keys
# @lru_cache(1024)
# def brand_check(brand: str,name: str):
#     # regularize name
#     name = regex.sub(lambda mo: name_dict[mo.string[mo.start():mo.end()]], name)
#     # check brand
#     if brand==MISSVALUE and name!=MISSVALUE:
#         words=name.split(" ")
#         for word in words:
#             if word in brand_dict:
#                 return brand_dict[word]
#     return brand
#
# def brand_fill(df: pd.DataFrame):
#     # fill brand
#     return df.apply(lambda x: brand_check(x.values[0],x.values[1]),axis=1)

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

def intersect_cnt(dfs: np.ndarray):
    intersect = np.empty(dfs.shape[0])
    for i in range(dfs.shape[0]):
        obs_tokens = set(dfs[i, 0].split(" "))
        target_tokens = set(dfs[i, 1].split(" "))
        intersect[i] = len(obs_tokens.intersection(target_tokens)) / (obs_tokens.__len__() + 1)
    return intersect

def get_extract_feature():
    def read_file(name: str):
        source = '../input/%s.tsv' % name
        df = pd.read_table(source, engine='c')
        return df

    def textclean(merge: pd.DataFrame):
        columns = ['name', 'category_name', 'brand_name', 'item_description']
        for col in columns: merge[col].fillna(value=MISSVALUE, inplace=True)

        columns = ['item_description', 'name','brand_name', 'category_name']
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
        p.close(); slices,dfvalue,dfs,df,p=None,None,None,None,None; gc.collect()
        return merge

    def inductive_brand(merge: pd.DataFrame):
        # inductive the missing brand from name
        print((merge['brand_name'] == MISSVALUE).sum())
        p = multiprocessing.Pool(THREAD)
        length = merge.shape[0]
        len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
        slices = [merge[:len1], merge[len1:len2], merge[len2:len3], merge[len3:]]
        dfvalue = []
        dfs = p.imap(brand_fill, slices)
        for df in dfs: dfvalue.append(df)
        # merge = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
        brand_name = pd.concat((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]), ignore_index=True)
        print((brand_name == MISSVALUE).sum())
        p.close(); slices,dfvalue,dfs,df,p=None,None,None,None,None; gc.collect()
        return brand_name

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
        dfAll['brand_name'] = inductive_brand(dfAll[['brand_name','name']])
        submission: pd.DataFrame = dftest[['test_id']]
        dftrain,dftest=None,None; gc.collect()
        return dfAll, submission

    def add_Frec_feat(dfAll: pd.DataFrame, col: str):
        s = dfAll[col].value_counts()
        s[MISSVALUE] = 0
        dfAll = dfAll.merge(s.to_frame(name=col + '_Frec'), left_on=col, right_index=True, how='left')
        s=None
        return dfAll

    dfAll, submission = get_cleaned_data()
    print('data cleaned')

    # add the Frec features
    columns = ['brand_name']
    print('add the Frec features')
    for col in columns: dfAll = add_Frec_feat(dfAll, col)

    # intersection count between 'item_description','name','brand_name','category_name'
    columns = [['brand_name', 'name'],['brand_name', 'item_description']]
    p = multiprocessing.Pool(THREAD)
    length = dfAll.shape[0]
    len1, len2, len3 = length // 4, length // 2, (length // 4) * 3
    for col in columns:
        slices = [dfAll[col].values[:len1], dfAll[col].values[len1:len2], dfAll[col].values[len2:len3], dfAll[col].values[len3:]]
        dfvalue = []
        dfs = p.imap(intersect_cnt, slices)
        for df in dfs: dfvalue.append(df)
        dfAll['%s_%s_Intsct' % (col[0], col[1])] = np.concatenate((dfvalue[0], dfvalue[1], dfvalue[2], dfvalue[3]))
    p.close(); slices,dfvalue,dfs,df,p=None,None,None,None,None; gc.collect()

    # count item_description length
    dfAll['item_description_wordLen'] = dfAll.item_description.apply(lambda x: len_splt(x))
    # nomalize price
    y_train = ((np.log1p(dfAll.price) - PRICE_MEAN) / PRICE_STD).values[:TRAIN_SIZE].reshape(-1, 1).astype(np.float32)
    dfAll.drop(['price', 'test_id', 'train_id'], axis=1, inplace=True)
    return dfAll, submission, y_train

def Preprocess_features(merge: pd.DataFrame, start_time):
    merge['item_condition_id'] = merge['item_condition_id'].astype('category')
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    # hand feature
    columns = ['brand_name_Frec', 'item_description_wordLen']
    for col in columns:
        merge[col] = np.log1p(merge[col])
        merge[col] = merge[col] / merge[col].max()

    # reduce memory
    for col in merge.columns:
        if str(merge[col].dtype)=='int64':
            merge[col]=merge[col].astype('int32')
        elif str(merge[col].dtype)=='float64':
            merge[col]=merge[col].astype('float32')

    hand_feature = ['brand_name_Frec',
                    'item_description_wordLen',
                    'brand_name_name_Intsct',
                    'brand_name_item_description_Intsct']
    return merge, hand_feature

def Vectorizors(df):
    df, name = df
    if name in ['category_2', 'category_name']:
        wb = CountVectorizer()
        return wb.fit_transform(df).astype(np.int32)
    if name in ['category_1', 'brand_name']:
        lb = LabelBinarizer(sparse_output=True)
        return lb.fit_transform(df).astype(np.int32)

def Get_Vectorizor(merge: pd.DataFrame):
    columns = ['category_1', 'category_2', 'category_name', 'brand_name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(Vectorizors, [[merge[col], col] for col in columns])
    results=[]
    for col, df in zip(columns, dfs): results.append(df)
    p.close(); dfs,df,p=None,None,None; gc.collect()
    return results[0],results[1],results[2],results[3]

def Split_Train_Test_FTRL(merge: pd.DataFrame, hand_feature, start_time):
    name_w1=param_space_best_WordBatch['name_w1']
    name_w2=param_space_best_WordBatch['name_w2']
    desc_w1=param_space_best_WordBatch['desc_w1']
    desc_w2=param_space_best_WordBatch['desc_w2']

    wb = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
        "hash_ngrams": 2,
        "hash_ngrams_weights": [name_w1, name_w2],
        "hash_size": 2 ** 28,
        "norm": None,
        "tf": 'binary',
        "idf": None,
    }), procs=8)
    wb.dictionary_freeze = True
    X_name = wb.fit_transform(merge['name']).astype(np.float32)
    del wb
    merge.drop(['name'], axis=1, inplace=True)
    X_name = X_name[:, np.array(np.clip(X_name[:TRAIN_SIZE].getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

    wb = wordbatch.WordBatch(normalize_text=None, extractor=(WordBag, {
        "hash_ngrams": 2,
        "hash_ngrams_weights": [desc_w1, desc_w2],
        "hash_size": 2 ** 28,
        "norm": "l2",
        "tf": 1.0,
        "idf": None
    }), procs=8)
    wb.dictionary_freeze = True
    X_description = wb.fit_transform(merge['item_description']).astype(np.float32)
    del wb
    merge.drop(['item_description'], axis=1, inplace=True)
    X_description = X_description[:, np.array(np.clip(X_description[:TRAIN_SIZE].getnnz(axis=0) - 1, 0, 1), dtype=bool)]
    print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

    X_category1, X_category2, X_category3, X_brand = Get_Vectorizor(merge)
    merge.drop(['category_1', 'category_2', 'category_name', 'brand_name'], axis=1, inplace=True)
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values.astype(np.float32))
    merge.drop(['item_condition_id', 'shipping'], axis=1, inplace=True)
    X_hand_feature = merge[hand_feature].values.astype(np.float32)
    merge.drop(hand_feature, axis=1, inplace=True)
    print('-'*50)

    # coo_matrix
    X_train = hstack((X_dummies[:TRAIN_SIZE], X_brand[:TRAIN_SIZE], X_category1[:TRAIN_SIZE],
                        X_category2[:TRAIN_SIZE], X_category3[:TRAIN_SIZE], X_hand_feature[:TRAIN_SIZE],
                        X_name[:TRAIN_SIZE], X_description[:TRAIN_SIZE]),dtype=np.float32)
    print('-'*50)
    X_test  = hstack((X_dummies[TRAIN_SIZE:], X_brand[TRAIN_SIZE:], X_category1[TRAIN_SIZE:],
                        X_category2[TRAIN_SIZE:], X_category3[TRAIN_SIZE:], X_hand_feature[TRAIN_SIZE:],
                        X_name[TRAIN_SIZE:], X_description[TRAIN_SIZE:]),dtype=np.float32)

    print(X_dummies.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_hand_feature.shape, X_name.shape, X_description.shape, X_train.shape, X_test.shape)
    X_dummies, X_brand, X_category1, X_category2, X_category3, X_hand_feature, X_name, X_description=None,\
    None,None,None,None,None,None,None
    gc.collect()

    # csr_matrix
    X_train = X_train.tocsr()
    print('[{}] X_train completed.'.format(time.time() - start_time))
    X_test = X_test.tocsr()
    print('[{}] X_test completed.'.format(time.time() - start_time))
    return X_train, X_test

class vanila_FM_FTRL_Regressor:
    def __init__(self, param_dict, D):
        alpha = param_dict['alpha']
        beta = param_dict['beta']
        L1 = param_dict['L1']
        L2 = param_dict['L2']
        alpha_fm = param_dict['alpha_fm']
        init_fm = param_dict['init_fm']
        D_fm = param_dict['D_fm']
        e_noise = param_dict['e_noise']
        iters = param_dict['iters']

        self.model = FM_FTRL(alpha=alpha,
                             beta=beta,
                             L1=L1,
                             L2=L2,
                             D=D,
                             alpha_fm=alpha_fm,
                             L2_fm=0.0,
                             init_fm=init_fm,
                             D_fm=D_fm,
                             e_noise=e_noise,
                             iters=iters,
                             inv_link="identity",
                             threads=THREAD)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

def main():
    start_time = time.time()
    merge, submission, y_train = get_extract_feature()
    print('[{}] data preparation done.'.format(time.time() - start_time))

    merge, hand_feature = Preprocess_features(merge, start_time)

    X_train, X_test = Split_Train_Test_FTRL(merge, hand_feature, start_time)
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))
    merge, hand_feature=None,None
    gc.collect()

    print('[{}] training FM_FTRL.'.format(time.time() - start_time))
    model = vanila_FM_FTRL_Regressor(param_space_best_FM_FTRL, D=X_train.shape[1])
    X_train=X_train.astype(np.float64)
    y_train=y_train.reshape(-1).astype(np.float64)
    model.fit(X_train, y_train)
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    X_train, y_train=None,None
    gc.collect()
    X_test=X_test.astype(np.float64)
    predsFM_FTRL = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    X_test, model=None,None
    gc.collect()

    submission['price'] = np.expm1(predsFM_FTRL * PRICE_STD + PRICE_MEAN)
    print(submission['price'].mean())
    submission.to_csv("submission_gru_lstm.csv", index=False)


if __name__ == "__main__":
    main()