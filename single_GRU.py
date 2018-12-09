# -*- coding: utf-8 -*-
# @Time    : 1/22/18 7:06 PM
# @Author  : LeeYun
# @File    : single_GRU.py
'''Description :

'''
import os,string,pickle,regex,time,gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fastcache import clru_cache as lru_cache
import multiprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend
from keras.backend import one_hot
from keras import optimizers
from keras.layers import Lambda

USE_GPU=False
TRAIN_SIZE = 1481661
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 50000
MISSVALUE='missvalue'

param_space_best_vanila_GRU = {
    'denselayer_units': 256,
    'description_Len': 90,
    'name_Len': 15,
    'embed_name': 30,
    'embed_desc': 60,
    'embed_brand': 10,
    'embed_cat_2': 10,
    'embed_cat_3': 10,
    'rnn_dim_name': 16,
    'rnn_dim_desc': 20,
    'lr': 0.0034832260328169426,
    'batch_size': 1200,
    'dense_drop': 0.0014241769141923634,
}

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

@lru_cache(1024)
def split_cat(text):
    text=text.split("/")
    if len(text)>=2:  return text[0],text[1]
    else: return text[0],MISSVALUE

@lru_cache(1024)
def len_splt(str,pat):
    return len(str.split(pat))

def text_processor(text):
    # save .&%$ 0-9 a-z , remove stop words
    text = text.str.lower(). \
        str.replace(r'[^\.&%$0-9a-z]', ' '). \
        str.replace(r'(\.|&|%|$|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4')
    return text.str.replace('\s+', ' ').str.strip()

def text_processor2(text):
    # save &/ 0-9 a-z
    text = text.str.lower(). \
        str.replace(r'[^0-9a-z&/]', ' '). \
        str.replace(r'(&|/|[0-9]+)', r' \1 ')
    return text.str.replace('\s+', ' ').str.strip()

def intersect_cnt(dfs):
    intersect=np.empty(dfs.shape[0])
    for i in range(dfs.shape[0]):
        obs_tokens=set(dfs[i,0].split(" "))
        target_tokens = set(dfs[i,1].split(" "))
        intersect[i]=len(obs_tokens.intersection(target_tokens))/(obs_tokens.__len__()+1)
    return intersect

def get_extract_feature():
    def read_file(name):
        source = '../input/%s.tsv' % name
        df = pd.read_table(source, engine='c')
        return df

    def textclean(dfAll: pd.DataFrame):
        '''clean name，category_name，brand_name，item_description'''
        def handle_missing_inplace(dataset):
            columns = ['name', 'category_name', 'brand_name', 'item_description']
            for col in columns: dataset[col].fillna(value=MISSVALUE, inplace=True)

        def rm_punctuation(merge):
            columns=['item_description','name'] #,'brand_name','category_name'
            p = multiprocessing.Pool(4)
            length=merge.shape[0]
            len1,len2,len3=length//4,length//2,(length//4)*3
            for col in columns:
                slices = [merge[col][:len1],merge[col][len1:len2],merge[col][len2:len3],merge[col][len3:]]
                dfvalue=[]
                dfs = p.imap(text_processor, slices)
                for df in dfs: dfvalue.append(df.values)
                merge[col] = np.concatenate((dfvalue[0],dfvalue[1],dfvalue[2],dfvalue[3]))

            columns=['brand_name','category_name']
            dfs = p.imap(text_processor2, [merge[col] for col in columns])
            for col, df in zip(columns, dfs): merge[col] = df
            merge['category_1'], merge['category_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))

        handle_missing_inplace(dfAll)
        rm_punctuation(dfAll)
        return dfAll

    def get_cleaned_data():
        dftrain=read_file('train')
        dftest=read_file('test')
        dftrain = dftrain[dftrain.price != 0]
        dfAll = pd.concat((dftrain, dftest), ignore_index=True)
        # dfAll=dfAll[:1000]
        dfAll = textclean(dfAll)
        submission: pd.DataFrame = dftest[['test_id']]
        return dfAll,submission

    def add_Frec_feat(dfAll,col):
        s=dfAll[col].value_counts()
        s[MISSVALUE] = 0
        dfAll = dfAll.merge(s.to_frame(name=col+'_Frec'), left_on=col, right_index=True, how='left')
        return dfAll

    dfAll,submission=get_cleaned_data()
    print('data cleaned')

    # add the Frec features
    columns = ['brand_name']
    print('add the Frec features')
    for col in columns: dfAll = add_Frec_feat(dfAll,col)

    # count item_description length
    dfAll['item_description_wordLen']=dfAll.item_description.apply(lambda x: len_splt(x,' '))

    # nomalize price
    dfAll['norm_price'] = (np.log1p(dfAll.price)-PRICE_MEAN)/PRICE_STD

    # intersection count between 'item_description','name','brand_name','category_name'
    columns = [['brand_name','name'],
              ['brand_name', 'item_description']]
    p = multiprocessing.Pool(4)
    dfs = p.imap(intersect_cnt, [dfAll[col].values for col in columns])
    for col, df in zip(columns, dfs): dfAll['%s_%s_Intsct'%(col[0],col[1])] = df

    return dfAll,submission

def Label_Encoder(df):
    le = LabelEncoder()
    return le.fit_transform(df)

def Item_Tokenizer(df):
    # do not concern the words only appeared in dftest
    tok_raw = Tokenizer(num_words=NumWords,filters='')
    tok_raw.fit_on_texts(df[:TRAIN_SIZE])
    return tok_raw.texts_to_sequences(df),min(tok_raw.word_counts.__len__(),NumWords)

def Preprocess_features(merge,start_time):
    merge['item_condition_id'] = merge['item_condition_id'].astype('category')
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    Item_size = {}
    # Label_Encoder brand_name + category
    columns = ['category_1', 'category_2', 'category_name', 'brand_name']
    p = multiprocessing.Pool(4)
    dfs = p.imap(Label_Encoder, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge[col] = df
        Item_size[col] = merge[col].max() + 1
    print('[{}] Label Encode `brand_name` and `categories` completed.'.format(time.time() - start_time))

    # sequance item_description,name
    columns = ['item_description', 'name']
    p = multiprocessing.Pool(4)
    dfs = p.imap(Item_Tokenizer, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge['Seq_' + col], Item_size[col] = df
    print('[{}] sequance `item_description` and `name` completed.'.format(time.time() - start_time))
    print(Item_size)

    # hand feature
    hand_feature = []
    for col in merge.columns:
        if ('Len' in col) or ('Frec' in col) or ('Intsct' in col):
            if ('Len' in col) or ('Frec' in col):
                merge[col] = np.log1p(merge[col])
                merge[col] = merge[col] / merge[col].max()
            hand_feature.append(col)
    print(hand_feature)

    return merge,Item_size,hand_feature

def Split_Train_Test(data, hand_feature):
    param_dict=param_space_best_vanila_GRU
    X_seq_item_description = pad_sequences(data['Seq_item_description'], maxlen=param_dict['description_Len'])
    X_seq_name = pad_sequences(data['Seq_name'], maxlen=param_dict['name_Len'])

    X_brand_name = data.brand_name.values.reshape(-1, 1)
    X_category_1 = data.category_1.values.reshape(-1, 1)
    X_category_2 = data.category_2.values.reshape(-1, 1)
    X_category_name = data.category_name.values.reshape(-1, 1)
    X_item_condition_id = (data.item_condition_id.values.astype(np.int32) - 1).reshape(-1, 1)
    X_shipping = (data.shipping.values - data.shipping.values.mean() / data.shipping.values.std()).reshape(-1, 1)
    X_hand_feature = (data[hand_feature].values - data[hand_feature].values.mean(axis=0)) / data[hand_feature].values.std(axis=0)

    X_train = dict(
        X_seq_item_description=X_seq_item_description[:TRAIN_SIZE],
        X_seq_name=X_seq_name[:TRAIN_SIZE],
        X_brand_name=X_brand_name[:TRAIN_SIZE],
        X_category_1=X_category_1[:TRAIN_SIZE],
        X_category_2=X_category_2[:TRAIN_SIZE],
        X_category_name=X_category_name[:TRAIN_SIZE],
        X_item_condition_id=X_item_condition_id[:TRAIN_SIZE],
        X_shipping=X_shipping[:TRAIN_SIZE],
        X_hand_feature=X_hand_feature[:TRAIN_SIZE],
    )
    X_valid = dict(
        X_seq_item_description=X_seq_item_description[TRAIN_SIZE:],
        X_seq_name=X_seq_name[TRAIN_SIZE:],
        X_brand_name=X_brand_name[TRAIN_SIZE:],
        X_category_1=X_category_1[TRAIN_SIZE:],
        X_category_2=X_category_2[TRAIN_SIZE:],
        X_category_name=X_category_name[TRAIN_SIZE:],
        X_item_condition_id=X_item_condition_id[TRAIN_SIZE:],
        X_shipping=X_shipping[TRAIN_SIZE:],
        X_hand_feature=X_hand_feature[TRAIN_SIZE:],
    )
    return X_train, X_valid

class vanila_GRU_Regressor:
    def __init__(self, param_dict,Item_size):
        if USE_GPU == False:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                    inter_op_parallelism_threads=4,
                                    allow_soft_placement=True, )
            session = tf.Session(config=config)
            backend.set_session(session)

        self.batch_size=int(param_dict['batch_size']*1.25)
        hand_feature_cols=len(Item_size['hand_feature'])

        name_seq_len=param_dict['name_Len']
        desc_seq_len=param_dict['description_Len']
        denselayer_units=param_dict['denselayer_units']
        embed_name=param_dict['embed_name']
        embed_desc=param_dict['embed_desc']
        embed_brand=param_dict['embed_brand']
        embed_cat_2=param_dict['embed_cat_2']
        embed_cat_3=param_dict['embed_cat_3']
        rnn_dim_name=param_dict['rnn_dim_name']
        rnn_dim_desc=param_dict['rnn_dim_desc']
        dense_drop=param_dict['dense_drop']

        name_voc_size=Item_size['name']
        desc_voc_size=Item_size['item_description']
        brand_voc_size=Item_size['brand_name']
        cat1_voc_size=Item_size['category_1']
        cat2_voc_size=Item_size['category_2']
        cat3_voc_size=Item_size['category_name']

        # Inputs
        X_seq_name = Input(shape=[name_seq_len], name="X_seq_name",dtype='int32')
        X_seq_item_description = Input(shape=[desc_seq_len],name="X_seq_item_description",dtype='int32')
        X_brand_name = Input(shape=[1], name="X_brand_name",dtype='int32')
        X_category_1 = Input(shape=[1], name="X_category_1",dtype='int32')
        X_category_2 = Input(shape=[1], name="X_category_2",dtype='int32')
        X_category_name = Input(shape=[1], name="X_category_name",dtype='int32')
        X_item_condition_id = Input(shape=[1], name="X_item_condition_id",dtype='uint8')
        X_shipping = Input(shape=[1], name="X_shipping",dtype='float32')
        X_hand_feature = Input(shape=[hand_feature_cols], name="X_hand_feature",dtype='float32')

        # Embeddings layers
        name = Embedding(name_voc_size, embed_name)(X_seq_name)
        item_desc = Embedding(desc_voc_size, embed_desc)(X_seq_item_description)
        brand = Embedding(brand_voc_size, embed_brand)(X_brand_name)
        cat_2 = Embedding(cat2_voc_size, embed_cat_2)(X_category_2)
        cat_3 = Embedding(cat3_voc_size, embed_cat_3)(X_category_name)

        # RNN layers
        name = GRU(rnn_dim_name)(name)
        item_desc = GRU(rnn_dim_desc)(item_desc)

        # OneHot layers
        cond = Lambda(one_hot, arguments={'num_classes': 5}, output_shape= (1,5))(X_item_condition_id)
        cat_1 = Lambda(one_hot, arguments={'num_classes': cat1_voc_size}, output_shape= (1,cat1_voc_size))(X_category_1)

        # main layer
        main_l = concatenate([
            name,
            item_desc,
            Flatten()(cat_1),
            Flatten()(cat_2),
            Flatten()(cat_3),
            Flatten()(brand),
            Flatten()(cond),
            X_shipping,
            X_hand_feature,
        ])
        main_l = Dropout(dense_drop)(Dense(denselayer_units, activation='relu')(main_l))
        output = Dense(1, activation="linear")(main_l)

        # model
        model = Model(
            [X_seq_name, X_seq_item_description, X_brand_name, X_category_1, X_category_2, X_category_name,
             X_item_condition_id, X_shipping, X_hand_feature],
            output)
        optimizer = optimizers.Adam(lr=param_dict['lr'])
        model.compile(loss='mse', optimizer=optimizer)
        self.model=model

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train, epochs=2, batch_size=self.batch_size,verbose=1)
        took = time.time() - t0
        print('took %.3fs' % (took))

    def predict(self,X_valid):
        return self.model.predict(X_valid,batch_size=self.batch_size).reshape(-1)

def main():
    start_time=time.time()

    # merge,submission = get_extract_feature()
    # _save('Data/final/data_preparation.pkl',[merge,submission])
    # # merge,submission=_load('Data/final/data_preparation.pkl')
    # y_train = merge['norm_price'].values[:TRAIN_SIZE].reshape(-1,1)
    # print('[{}] data preparation done.'.format(time.time() - start_time))
    #
    # merge,Item_size,hand_feature=Preprocess_features(merge,start_time)
    # _save('Data/final/feature_process.pkl',[merge,Item_size,hand_feature])
    # # merge, Item_size, hand_feature=_load('Data/final/feature_process.pkl')
    # Item_size['hand_feature']=hand_feature
    #
    # _save('Data/final/Split_Train_Test2.pkl',[submission,Item_size])
    submission,Item_size=_load('Data/final/Split_Train_Test2.pkl')
    print('[{}] [submission,Item_size] saved'.format(time.time() - start_time))

    # X_train,X_test=Split_Train_Test(merge, hand_feature)
    # _save('Data/final/Split_Train_Test.pkl',[X_train,X_test,y_train])
    X_train, X_test, y_train=_load('Data/final/Split_Train_Test.pkl')
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))
    # del merge,hand_feature
    # gc.collect()

    print('[{}] training conv1d.'.format(time.time() - start_time))
    model=vanila_GRU_Regressor(param_space_best_vanila_GRU,Item_size)
    model.fit(X_train, y_train)
    print('[{}] Train conv1d completed'.format(time.time() - start_time))
    VALID_SIZE=submission.shape[0]
    predsConv1d=model.predict(X_test)
    print('[{}] Predict conv1d completed'.format(time.time() - start_time))
    del X_train,y_train,X_test,model,Item_size
    gc.collect()

    submission['price'] = np.expm1(predsConv1d*PRICE_STD + PRICE_MEAN)
    print(submission['price'].mean())
    submission.to_csv("submission_gru_lstm.csv", index=False)

if __name__ == "__main__":
    main()






















































