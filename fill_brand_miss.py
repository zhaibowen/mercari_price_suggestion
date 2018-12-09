# -*- coding: utf-8 -*-
# @Time    : 1/31/18 2:56 PM
# @Author  : LeeYun
# @File    : fill_brand_miss.py
'''Description :

'''

import time,gc,re,string,multiprocessing,pickle
import pandas as pd
import numpy as np
from fastcache import clru_cache as lru_cache

TRAIN_SIZE = 1481661
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 70000
THREAD = 4
MISSVALUE = 'missvalue'

import re


# def multiple_replace(dict, text):
#   # Create a regular expression  from the dictionary keys
#   regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
#
#   # For each match, look-up corresponding value in dictionary
#   return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)
#
# if __name__ == "__main__":
#
#   text = "Larry Wall is the creator of Perl"
#
#   dict = {
#     "Larry Wall" : "Guido van Rossum",
#     "creator" : "Benevolent Dictator for Life",
#     "Perl" : "Python",
#   }
#
#   print(multiple_replace(dict, text))


#
# brand_keywords=frozenset(['ipad','ipod','apple','macbook','iphone'])
# brand_list=(
#     (['ipad','ipod','apple','macbook','iphone'], 'apple'),
# )
# brand_dict={}
# for keywords,brand in brand_list:
#     for item in keywords:
#         brand_dict[item]=brand
# a=1

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

def text_processor(text: pd.Series):
    return text.str.lower(). \
        str.replace(r'([a-z]+|[0-9]+|[^0-9a-z])', r' \1 '). \
        str.replace(r'([0-9])( *)\.( *)([0-9])', r'\1.\4').\
        str.replace('\s+', ' ').\
        str.strip()

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

# dftrain = pd.read_table('Data/train.tsv', engine='c')
# dftest = pd.read_table('Data/test.tsv', engine='c')
# dftrain = dftrain[dftrain.price != 0]
# dfAll = pd.concat((dftrain, dftest), ignore_index=True)
# dfAll = textclean(dfAll)
# _save('Data/textclean', dfAll)
dfAll=_load('Data/textclean')
dfAll=dfAll[['brand_name', 'name']]


####################################################################
# brand_frec=dfAll['brand_name'].value_counts()
# with open("Data/brand_frec.txt", "w") as f:
#     for i in range(brand_frec.shape[0]):
#         f.write('%s\t\t\t%d\n'%(brand_frec.index.values[i],brand_frec.values[i]))

####################################################################
# brand_frec=dfAll['brand_name'].value_counts()
# with open("Data/brand_name_list.txt", "w") as f:
#     for i in range(brand_frec.shape[0]):
#         text=brand_frec.index.values[i]
#         text2='_'.join(text.split(' '))
#         if text!=text2:
#             f.write('(["%s"],"%s"),\n'%(text,text2))

####################################################################
# brand_frec=dfAll['brand_name'].value_counts()
# with open("Data/brand_brand_list.txt", "w") as f:
#     for i in range(brand_frec.shape[0]):
#         text=brand_frec.index.values[i]
#         text2='_'.join(text.split(' '))
#         f.write('(["%s"],"%s"),\n'%(text2,text))

# all_brand=dfAll['brand_name'].unique()
# print((dfAll['brand_name']==MISSVALUE).sum())
# dfAll_null_brand=dfAll.loc[dfAll['brand_name']==MISSVALUE]

lularoe=dfAll.loc[dfAll['brand_name']=='victoria \' s secret']

a=1




























































