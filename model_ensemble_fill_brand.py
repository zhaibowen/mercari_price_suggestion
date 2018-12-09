# -*- coding: utf-8 -*-
# @Time    : 1/25/18 5:03 PM
# @Author  : LeeYun
# @File    : model_ensemble.py
'''Description :
0.403
'''
import os, string, pickle, re, time, gc, multiprocessing, wordbatch
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, hstack
from fastcache import clru_cache as lru_cache
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras import backend, optimizers
from keras.models import Model
from keras.backend import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Lambda
from wordbatch.models import FM_FTRL
from wordbatch.extractors import WordBag

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DEVELOP = False
EPOCH = 2

if DEVELOP:
    TRAIN_SIZE = 10000
else:
    TRAIN_SIZE = 1481661
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 50000
THREAD = 4
MISSVALUE = 'missvalue'

# 0.42192
param_space_best_vanila_con1d = {
    'denselayer_units': 244,
    'description_Len': 85,
    'name_Len': 10,
    'embed_name': 58,
    'embed_desc': 52,
    'embed_brand': 22,
    'embed_cat_2': 14,
    'embed_cat_3': 37,
    'name_filter': 114,
    'desc_filter': 96,
    'name_filter_size': 4,
    'desc_filter_size': 4,
    'lr': 0.003924230325700921,
    'batch_size': 933,
    'dense_drop': 0.009082372998548981,
}

# 0.42368
param_space_best_vanila_GRU = {
    'denselayer_units': 232,
    'description_Len': 85,
    'name_Len': 10,
    'embed_name': 49,
    'embed_desc': 56,
    'embed_brand': 29,
    'embed_cat_2': 12,
    'embed_cat_3': 37,
    'rnn_dim_name': 23,
    'rnn_dim_desc': 24,
    'lr': 0.0036379197189225143,
    'batch_size': 597,
    'dense_drop': 0.00498597880773485,
}

# 0.41250
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

param_space_best_WordBatch = {
    'desc_w1': 1.3740067995315037,
    'desc_w2': 1.0248685266832964,
    'name_w1': 2.1385527373939834,
    'name_w2': 0.3894761681383836,
}

param_space_best_ensemble = {
    'GRU_weight': 0.45241912269307594,
    'conv1d_weight': 0.42898887735589847,
    'FM_FTRL_weight': 0.9788102443866593,
}
ensemble_total_weights = param_space_best_ensemble['GRU_weight'] + param_space_best_ensemble['conv1d_weight'] + \
                         param_space_best_ensemble['FM_FTRL_weight']
param_space_best_ensemble['GRU_weight'] /= ensemble_total_weights
param_space_best_ensemble['conv1d_weight'] /= ensemble_total_weights
param_space_best_ensemble['FM_FTRL_weight'] /= ensemble_total_weights
print(param_space_best_ensemble)

# regularize name
name_list=(
    (['lebron soldier'], 'lebron_soldier'),
    (['air force','air forces'], 'air_force'),
    (['air max'], 'air_max'),
    (['vs pink'], 'vs_pink'),
    (['victoria s secret','victoria secret','victoria secrets','v . s .'], 'victoria_secret'),
    (['forever 21','f 21'], 'forever_21'),
    (['game boy'], 'game_boy'),
    (['lulu lemon','lulu'], 'lululemon'),
    (['michael kors'], 'michael_kors'),
    (['american eagle','ae','a . e .'], 'american_eagle'),
    (['rae dunn'], 'rae_dunn'),(["bath & body works"],"bath_&_body_works"),
    (["under armour"],"under_armour"),
    (["old navy"],"old_navy"),
    (["carter s"],"carter_s"),
    (["the north face"],"the_north_face"),
    (["urban decay"],"urban_decay"),
    (["too faced"],"too_faced"),
    (["brandy melville"],"brandy_melville"),
    (["kate spade"],"kate_spade"),
    (["kendra scott"],"kendra_scott"),
    (["ugg australia"],"ugg_australia"),
    (["polo ralph lauren"],"polo_ralph_lauren"),
    (["charlotte russe"],"charlotte_russe"),
    (["vera bradley"],"vera_bradley"),
    (["ralph lauren"],"ralph_lauren"),
    (["h & m"],"h_&_m"),
    (["tory burch"],"tory_burch"),
    (["free people"],"free_people"),
    (["air jordan"],"air_jordan"),
    (["miss me"],"miss_me"),
    (["louis vuitton"],"louis_vuitton"),
    (["abercrombie & fitch"],"abercrombie_&_fitch"),
    (["hot topic"],"hot_topic"),
    (["lilly pulitzer"],"lilly_pulitzer"),
    (["calvin klein"],"calvin_klein"),
    (["levi s"],"levi_s"),
    (["kylie cosmetics"],"kylie_cosmetics"),
    (["mary kay"],"mary_kay"),
    (["american boy & girl"],"american_boy_&_girl"),
    (["anastasia beverly hills"],"anastasia_beverly_hills"),
    (["tommy hilfiger"],"tommy_hilfiger"),
    (["steve madden"],"steve_madden"),
    (["american girl"],"american_girl"),
    (["kat von d"],"kat_von_d"),
    (["customized & personalized"],"customized_&_personalized"),
    (["alex and ani"],"alex_and_ani"),
    (["urban outfitters"],"urban_outfitters"),
    (["ray ban"],"ray_ban"),
    (["betsey johnson"],"betsey_johnson"),
    (["rock revival"],"rock_revival"),
    (["harley davidson"],"harley_davidson"),
    (["vineyard vines"],"vineyard_vines"),
    (["dooney & bourke"],"dooney_&_bourke"),
    (["juicy couture"],"juicy_couture"),
    (["true religion brand jeans"],"true_religion_brand_jeans"),
)
# map from keyword to brand
brand_list=(
    (['nike','lebron_soldier','kyrie','air_force','air_max'],'nike'),
    (['vs_pink'],'pink'),
    (['victoria_secret','vsx'],'victoria s secret'),
    (['lularoe','llr'],'lularoe'),
    (['ipad','ipod','apple','macbook','iphone'], 'apple'),
    (['forever_21'], 'forever 21'),
    (['nintendo', 'wii', 'gameboy', 'zelda', 'gamecube',],'nintendo'),
    (['lululemon'],'lululemon'),
    (['michael_kors'],'michael kors'),
    (['american_eagle','aeo','aerie'],'american eagle'),
    (['rae_dunn'],'rae dunn'),
    (['disney', 'disneyland', 'minnie', 'mickey', 'woody', 'tsum'],'disney'),
    (["sephora"],"sephora"),
    (["coach"],"coach"),
    (["bath_&_body_works"],"bath & body works"),
    (["adidas"],"adidas"),
    (["funko"],"funko"),
    (["under_armour"],"under armour"),
    (["sony"],"sony"),
    (["old_navy"],"old navy"),
    (["hollister"],"hollister"),
    (["carter_s"],"carter s"),
    (["the_north_face"],"the north face"),
    (["urban_decay"],"urban decay"),
    (["independent"],"independent"),
    (["too_faced"],"too faced"),
    (["xbox"],"xbox"),
    (["brandy_melville"],"brandy melville"),
    (["kate_spade"],"kate spade"),
    (["mac"],"mac"),
    (["gap"],"gap"),
    (["kendra_scott"],"kendra scott"),
    (["tarte"],"tarte"),
    (["ugg_australia"],"ugg australia"),
    (["vans"],"vans"),
    (["polo_ralph_lauren"],"polo ralph lauren"),
    (["charlotte_russe"],"charlotte russe"),
    (["vera_bradley"],"vera bradley"),
    (["samsung"],"samsung"),
    (["senegence"],"senegence"),
    (["converse"],"converse"),
    (["ralph_lauren"],"ralph lauren"),
    (["h_&_m"],"h & m"),
    (["tory_burch"],"tory burch"),
    (["free_people"],"free people"),
    (["air_jordan"],"air jordan"),
    (["miss_me"],"miss me"),
    (["louis_vuitton"],"louis vuitton"),
    (["express"],"express"),
    (["abercrombie_&_fitch"],"abercrombie & fitch"),
    (["nyx"],"nyx"),
    (["pokemon"],"pokemon"),
    (["hot_topic"],"hot topic"),
    (["lilly_pulitzer"],"lilly pulitzer"),
    (["calvin_klein"],"calvin klein"),
    (["levi_s"],"levi s"),
    (["kylie_cosmetics"],"kylie cosmetics"),
    (["pandora"],"pandora"),
    (["mary_kay"],"mary kay"),
    (["american_boy_&_girl"],"american boy & girl"),
    (["anastasia_beverly_hills"],"anastasia beverly hills"),
    (["benefit"],"benefit"),
    (["torrid"],"torrid"),
    (["chanel"],"chanel"),
    (["tommy_hilfiger"],"tommy hilfiger"),
    (["steve_madden"],"steve madden"),
    (["scentsy"],"scentsy"),
    (["aeropostale"],"aeropostale"),
    (["mossimo"],"mossimo"),
    (["vintage"],"vintage"),
    (["columbia"],"columbia"),
    (["american_girl"],"american girl"),
    (["fitbit"],"fitbit"),
    (["kat_von_d"],"kat von d"),
    (["bebe"],"bebe"),
    (["customized_&_personalized"],"customized & personalized"),
    (["guess"],"guess"),
    (["alex_and_ani"],"alex and ani"),
    (["urban_outfitters"],"urban outfitters"),
    (["ray_ban"],"ray ban"),
    (["gucci"],"gucci"),
    (["betsey_johnson"],"betsey johnson"),
    (["jordan"],"jordan"),
    (["rock_revival"],"rock revival"),
    (["target"],"target"),
    (["patagonia"],"patagonia"),
    (["clinique"],"clinique"),
    (["harley_davidson"],"harley davidson"),
    (["mattel"],"mattel"),
    (["reebok"],"reebok"),
    (["gymboree"],"gymboree"),
    (["vineyard_vines"],"vineyard vines"),
    (["dooney_&_bourke"],"dooney & bourke"),
    (["xhilaration"],"xhilaration"),
    (["puma"],"puma"),
    (["maybelline"],"maybelline"),
    (["juicy_couture"],"juicy couture"),
    (["nfl"],"nfl"),
    (["true_religion_brand_jeans"],"true religion brand jeans"),
)
name_dict={}
for keywords,value in name_list:
    for item in keywords:
        name_dict[item]=value#+'** '
brand_dict={}
for keywords,brand in brand_list:
    for item in keywords:
        brand_dict[item]=brand#+'** '
del name_list,brand_list

# regularize name
regex = re.compile("(%s)" % "|".join(map(re.escape, name_dict.keys())))
# Create a regular expression  from the dictionary keys
@lru_cache(1024)
def brand_check(brand: str,name: str):
    # regularize name
    name = regex.sub(lambda mo: name_dict[mo.string[mo.start():mo.end()]], name)
    # check brand
    if brand==MISSVALUE and name!=MISSVALUE:
        words=name.split(" ")
        for word in words:
            if word in brand_dict:
                return brand_dict[word]
    return brand

def brand_fill(df: pd.DataFrame):
    # fill brand
    return df.apply(lambda x: brand_check(x.values[0],x.values[1]),axis=1)


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


def prepare_batches(seq: np.ndarray, step: int):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i + step])
    return res


def text_processor(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'[^\.!?@#&%$/\\0-9a-z]', ' '). \
        str.replace(r'(\.|!|\?|@|#|&|%|\$|/|\\|[0-9]+)', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4'). \
        str.replace('\s+', ' '). \
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
        if DEVELOP: dfAll = dfAll[:20000]
        dfAll = textclean(dfAll)
        dfAll['brand_name'] = inductive_brand(dfAll[['brand_name','name']])
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


def Preprocess_features(merge: pd.DataFrame, start_time):
    merge['item_condition_id'] = merge['item_condition_id'].astype('category')
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    Item_size = {}
    # Label_Encoder brand_name + category
    columns = ['category_1', 'category_2', 'category_name', 'brand_name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(Label_Encoder, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge['Lb_' + col] = df
        Item_size[col] = merge['Lb_' + col].max() + 1
    print('[{}] Label Encode `brand_name` and `categories` completed.'.format(time.time() - start_time))
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()

    # sequance item_description,name
    columns = ['item_description', 'name']
    p = multiprocessing.Pool(THREAD)
    dfs = p.imap(Item_Tokenizer, [merge[col] for col in columns])
    for col, df in zip(columns, dfs):
        merge['Seq_' + col], Item_size[col] = df
    print('[{}] sequance `item_description` and `name` completed.'.format(time.time() - start_time))
    print(Item_size)
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()

    # hand feature
    columns = ['brand_name_Frec', 'item_description_wordLen']
    for col in columns:
        merge[col] = np.log1p(merge[col])
        merge[col] = merge[col] / merge[col].max()

    # reduce memory
    for col in merge.columns:
        if str(merge[col].dtype) == 'int64':
            merge[col] = merge[col].astype('int32')
        elif str(merge[col].dtype) == 'float64':
            merge[col] = merge[col].astype('float32')

    hand_feature = ['brand_name_Frec',
                    'item_description_wordLen',
                    'brand_name_name_Intsct',
                    'brand_name_item_description_Intsct']
    return merge, Item_size, hand_feature


def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup


def conv1d(inputs, num_filters, filter_size, padding='same', strides=1):
    he_std = np.sqrt(2 / (filter_size * num_filters))
    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        strides=strides,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out


def dense(X, size, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    out = tf.layers.dense(X, units=size, activation=activation,
                          kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out


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
    results = []
    for col, df in zip(columns, dfs): results.append(df)
    p.close();
    dfs, df, p = None, None, None;
    gc.collect()
    return results[0], results[1], results[2], results[3]


def Split_Train_Test_NN(data: pd.DataFrame, hand_feature):
    param_dict = param_space_best_vanila_con1d
    X_seq_item_description = pad_sequences(data['Seq_item_description'], maxlen=param_dict['description_Len'])
    X_seq_name = pad_sequences(data['Seq_name'], maxlen=param_dict['name_Len'])

    X_brand_name = data.Lb_brand_name.values.reshape(-1, 1)
    X_category_1 = data.Lb_category_1.values.reshape(-1, 1)
    X_category_2 = data.Lb_category_2.values.reshape(-1, 1)
    X_category_name = data.Lb_category_name.values.reshape(-1, 1)
    X_item_condition_id = (data.item_condition_id.values.astype(np.int32) - 1).reshape(-1, 1)
    X_shipping = ((data.shipping.values - data.shipping.values.mean()) / data.shipping.values.std()).reshape(-1, 1)
    X_hand_feature = (data[hand_feature].values - data[hand_feature].values.mean(axis=0)) / data[
        hand_feature].values.std(axis=0)

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
    X_seq_item_description, X_seq_name, X_brand_name, X_category_1, X_category_2, X_category_name, \
    X_item_condition_id, X_shipping, X_hand_feature = None, None, None, None, None, None, None, None, None
    return X_train, X_valid


def Split_Train_Test_FTRL(merge: pd.DataFrame, hand_feature, start_time):
    desc_w1 = param_space_best_WordBatch['desc_w1']
    desc_w2 = param_space_best_WordBatch['desc_w2']
    name_w1 = param_space_best_WordBatch['name_w1']
    name_w2 = param_space_best_WordBatch['name_w2']

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

    X_dummies = csr_matrix(
        pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values.astype(np.float32))
    merge.drop(['item_condition_id', 'shipping'], axis=1, inplace=True)
    X_hand_feature = merge[hand_feature].values.astype(np.float32)
    merge.drop(hand_feature, axis=1, inplace=True)
    print('-' * 50)

    # coo_matrix
    X_train = hstack((X_dummies[:TRAIN_SIZE], X_brand[:TRAIN_SIZE], X_category1[:TRAIN_SIZE],
                      X_category2[:TRAIN_SIZE], X_category3[:TRAIN_SIZE], X_hand_feature[:TRAIN_SIZE],
                      X_name[:TRAIN_SIZE], X_description[:TRAIN_SIZE]), dtype=np.float32)
    print('-' * 50)
    X_test = hstack((X_dummies[TRAIN_SIZE:], X_brand[TRAIN_SIZE:], X_category1[TRAIN_SIZE:],
                     X_category2[TRAIN_SIZE:], X_category3[TRAIN_SIZE:], X_hand_feature[TRAIN_SIZE:],
                     X_name[TRAIN_SIZE:], X_description[TRAIN_SIZE:]), dtype=np.float32)

    print(X_dummies.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_hand_feature.shape, X_name.shape, X_description.shape, X_train.shape, X_test.shape)
    X_dummies, X_brand, X_category1, X_category2, X_category3, X_hand_feature, X_name, X_description = None, \
                                                                                                       None, None, None, None, None, None, None
    gc.collect()

    # csr_matrix
    X_train = X_train.tocsr()
    print('[{}] X_train completed.'.format(time.time() - start_time))
    X_test = X_test.tocsr()
    print('[{}] X_test completed.'.format(time.time() - start_time))
    return X_train, X_test


class vanila_conv1d_Regressor:
    def __init__(self, param_dict, Item_size):
        self.seed = 2018
        self.batch_size = int(param_dict['batch_size'] * 1.25)
        self.lr = param_dict['lr']

        name_seq_len = param_dict['name_Len']
        desc_seq_len = param_dict['description_Len']
        denselayer_units = param_dict['denselayer_units']
        embed_name = param_dict['embed_name']
        embed_desc = param_dict['embed_desc']
        embed_brand = param_dict['embed_brand']
        embed_cat_2 = param_dict['embed_cat_2']
        embed_cat_3 = param_dict['embed_cat_3']
        name_filter = param_dict['name_filter']
        desc_filter = param_dict['desc_filter']
        name_filter_size = param_dict['name_filter_size']
        desc_filter_size = param_dict['desc_filter_size']
        dense_drop = param_dict['dense_drop']

        name_voc_size = Item_size['name']
        desc_voc_size = Item_size['item_description']
        brand_voc_size = Item_size['brand_name']
        cat1_voc_size = Item_size['category_1']
        cat2_voc_size = Item_size['category_2']
        cat3_voc_size = Item_size['category_name']

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.as_default():
            self.place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
            self.place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
            self.place_brand = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat1 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat2 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_cat3 = tf.placeholder(tf.int32, shape=(None, 1))
            self.place_ship = tf.placeholder(tf.float32, shape=(None, 1))
            self.place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
            self.place_hand = tf.placeholder(tf.float32, shape=(None, len(Item_size['hand_feature'])))

            self.place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            self.place_lr = tf.placeholder(tf.float32, shape=(), )

            self.is_train = tf.placeholder(tf.bool, shape=(), )

            name = embed(self.place_name, name_voc_size, embed_name)
            desc = embed(self.place_desc, desc_voc_size, embed_desc)
            brand = embed(self.place_brand, brand_voc_size, embed_brand)
            cat_2 = embed(self.place_cat2, cat2_voc_size, embed_cat_2)
            cat_3 = embed(self.place_cat3, cat3_voc_size, embed_cat_3)

            name = conv1d(name, num_filters=name_filter, filter_size=name_filter_size)
            name = tf.layers.average_pooling1d(name, pool_size=int(name.shape[1]), strides=1, padding='valid')
            name = tf.contrib.layers.flatten(name)

            desc = conv1d(desc, num_filters=desc_filter, filter_size=desc_filter_size)
            desc = tf.layers.average_pooling1d(desc, pool_size=int(desc.shape[1]), strides=1, padding='valid')
            desc = tf.contrib.layers.flatten(desc)

            brand = tf.contrib.layers.flatten(brand)

            cat_1 = tf.one_hot(self.place_cat1, cat1_voc_size)
            cat_1 = tf.contrib.layers.flatten(cat_1)

            cat_2 = tf.contrib.layers.flatten(cat_2)
            cat_3 = tf.contrib.layers.flatten(cat_3)

            hand_feat = self.place_hand
            ship = self.place_ship

            cond = tf.one_hot(self.place_cond, 5)
            cond = tf.contrib.layers.flatten(cond)

            out = tf.concat([name, desc, brand, cat_1, cat_2, cat_3, ship, cond, hand_feat], axis=1)
            out = dense(out, denselayer_units, activation=tf.nn.relu)
            out = tf.layers.dropout(out, rate=dense_drop, training=self.is_train)
            self.out = dense(out, 1)

            loss = tf.losses.mean_squared_error(self.place_y, self.out)
            opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
            self.train_step = opt.minimize(loss)

            init = tf.global_variables_initializer()

        config = tf.ConfigProto(intra_op_parallelism_threads=THREAD,
                                inter_op_parallelism_threads=THREAD,
                                allow_soft_placement=True, )
        self.session = tf.Session(config=config, graph=graph)
        self.init = init

    def fit(self, X_train, y_train):
        self.session.run(self.init)
        for epoch in range(EPOCH):
            np.random.seed(epoch)

            train_idx_shuffle = np.arange(TRAIN_SIZE)
            np.random.shuffle(train_idx_shuffle)
            batches = prepare_batches(train_idx_shuffle, self.batch_size)

            for rnd, idx in enumerate(batches):
                # if rnd%100==0:
                #     print(rnd*self.batch_size)
                feed_dict = {
                    self.place_name: X_train['X_seq_name'][idx],
                    self.place_desc: X_train['X_seq_item_description'][idx],
                    self.place_brand: X_train['X_brand_name'][idx],
                    self.place_cat1: X_train['X_category_1'][idx],
                    self.place_cat2: X_train['X_category_2'][idx],
                    self.place_cat3: X_train['X_category_name'][idx],
                    self.place_cond: X_train['X_item_condition_id'][idx],
                    self.place_ship: X_train['X_shipping'][idx],
                    self.place_hand: X_train['X_hand_feature'][idx],
                    self.place_y: y_train[idx],
                    self.place_lr: self.lr,
                    self.is_train: True,
                }
                self.session.run(self.train_step, feed_dict=feed_dict)

    def predict(self, X_valid, valid_len):
        y_pred = np.zeros(valid_len)
        test_idx = np.arange(valid_len)
        batches = prepare_batches(test_idx, self.batch_size)

        for idx in batches:
            feed_dict = {
                self.place_name: X_valid['X_seq_name'][idx],
                self.place_desc: X_valid['X_seq_item_description'][idx],
                self.place_brand: X_valid['X_brand_name'][idx],
                self.place_cat1: X_valid['X_category_1'][idx],
                self.place_cat2: X_valid['X_category_2'][idx],
                self.place_cat3: X_valid['X_category_name'][idx],
                self.place_cond: X_valid['X_item_condition_id'][idx],
                self.place_ship: X_valid['X_shipping'][idx],
                self.place_hand: X_valid['X_hand_feature'][idx],
                self.is_train: False,
            }
            batch_pred = self.session.run(self.out, feed_dict=feed_dict)
            y_pred[idx] = batch_pred[:, 0]

        self.session.close()
        return y_pred


class vanila_GRU_Regressor:
    def __init__(self, param_dict, Item_size):
        config = tf.ConfigProto(intra_op_parallelism_threads=THREAD,
                                inter_op_parallelism_threads=THREAD,
                                allow_soft_placement=True, )
        session = tf.Session(config=config)
        backend.set_session(session)

        self.batch_size = int(param_dict['batch_size'] * 1.25)
        hand_feature_cols = len(Item_size['hand_feature'])

        name_seq_len = param_dict['name_Len']
        desc_seq_len = param_dict['description_Len']
        denselayer_units = param_dict['denselayer_units']
        embed_name = param_dict['embed_name']
        embed_desc = param_dict['embed_desc']
        embed_brand = param_dict['embed_brand']
        embed_cat_2 = param_dict['embed_cat_2']
        embed_cat_3 = param_dict['embed_cat_3']
        rnn_dim_name = param_dict['rnn_dim_name']
        rnn_dim_desc = param_dict['rnn_dim_desc']
        dense_drop = param_dict['dense_drop']

        name_voc_size = Item_size['name']
        desc_voc_size = Item_size['item_description']
        brand_voc_size = Item_size['brand_name']
        cat1_voc_size = Item_size['category_1']
        cat2_voc_size = Item_size['category_2']
        cat3_voc_size = Item_size['category_name']

        # Inputs
        X_seq_name = Input(shape=[name_seq_len], name="X_seq_name", dtype='int32')
        X_seq_item_description = Input(shape=[desc_seq_len], name="X_seq_item_description", dtype='int32')
        X_brand_name = Input(shape=[1], name="X_brand_name", dtype='int32')
        X_category_1 = Input(shape=[1], name="X_category_1", dtype='int32')
        X_category_2 = Input(shape=[1], name="X_category_2", dtype='int32')
        X_category_name = Input(shape=[1], name="X_category_name", dtype='int32')
        X_item_condition_id = Input(shape=[1], name="X_item_condition_id", dtype='uint8')
        X_shipping = Input(shape=[1], name="X_shipping", dtype='float32')
        X_hand_feature = Input(shape=[hand_feature_cols], name="X_hand_feature", dtype='float32')

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
        cond = Lambda(one_hot, arguments={'num_classes': 5}, output_shape=(1, 5))(X_item_condition_id)
        cat_1 = Lambda(one_hot, arguments={'num_classes': cat1_voc_size}, output_shape=(1, cat1_voc_size))(X_category_1)

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
        self.model = model

    def fit(self, X_train, y_train):
        t0 = time.time()
        self.model.fit(X_train, y_train, epochs=EPOCH, batch_size=self.batch_size, verbose=2)
        took = time.time() - t0
        print('took %.3fs' % (took))

    def predict(self, X_valid):
        pred = self.model.predict(X_valid, batch_size=self.batch_size).reshape(-1)
        backend.clear_session()
        return pred


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

    if DEVELOP:
        VALID_SIZE = 10000
    else:
        VALID_SIZE = submission.shape[0]

    merge, Item_size, hand_feature = Preprocess_features(merge, start_time)
    Item_size['hand_feature'] = hand_feature

    X_train, X_test = Split_Train_Test_NN(merge, hand_feature)
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))

    print('[{}] training conv1d.'.format(time.time() - start_time))
    model = vanila_conv1d_Regressor(param_space_best_vanila_con1d, Item_size)
    model.fit(X_train, y_train)
    print('[{}] Train conv1d completed'.format(time.time() - start_time))
    predsConv1d = model.predict(X_test, VALID_SIZE).astype(np.float32)
    print('[{}] Predict conv1d completed'.format(time.time() - start_time))

    print('[{}] training GRU.'.format(time.time() - start_time))
    model = vanila_GRU_Regressor(param_space_best_vanila_GRU, Item_size)
    model.fit(X_train, y_train)
    print('[{}] Train GRU completed'.format(time.time() - start_time))
    predsGRU = model.predict(X_test).astype(np.float32)
    print('[{}] Predict GRU completed'.format(time.time() - start_time))
    X_train, X_test, model, Item_size, VALID_SIZE = None, None, None, None, None
    merge.drop(['Lb_category_1', 'Lb_category_2', 'Lb_category_name',
                'Lb_brand_name', 'Seq_item_description', 'Seq_name'],
               axis=1, inplace=True)
    gc.collect()

    X_train, X_test = Split_Train_Test_FTRL(merge, hand_feature, start_time)
    print('[{}] Split_Train_Test completed'.format(time.time() - start_time))
    merge, hand_feature = None, None
    gc.collect()

    print('[{}] training FM_FTRL.'.format(time.time() - start_time))
    model = vanila_FM_FTRL_Regressor(param_space_best_FM_FTRL, D=X_train.shape[1])
    X_train = X_train.astype(np.float64)
    y_train = y_train.reshape(-1).astype(np.float64)
    model.fit(X_train, y_train)
    print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))
    X_train, y_train = None, None
    gc.collect()
    X_test = X_test.astype(np.float64)
    predsFM_FTRL = model.predict(X_test)
    print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    X_test, model = None, None
    gc.collect()

    preds = param_space_best_ensemble['FM_FTRL_weight'] * predsFM_FTRL + \
            param_space_best_ensemble['conv1d_weight'] * predsConv1d + \
            param_space_best_ensemble['GRU_weight'] * predsGRU

    submission['price'] = np.expm1(preds * PRICE_STD + PRICE_MEAN)
    print(submission['price'].mean())
    submission.to_csv("submission_gru_lstm.csv", index=False)


if __name__ == "__main__":
    main()