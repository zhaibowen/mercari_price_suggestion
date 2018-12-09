# -*- coding: utf-8 -*-
# @Time    : 1/15/18 9:03 AM
# @Author  : LeeYun
# @File    : main_0111_gru.py
'''Description :

'''
import os,numba,string,pickle,nltk,time,time_utils,logging_utils,gc,re
import pandas as pd
import numpy as np
from config import *
from matplotlib import pyplot as plt
from vanila_FTRL_utils import vanila_WorldBatch_Regressor
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

THREAD = 4

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
def len_splt(str: str):
    return len(str.split(' '))

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
        merge['has_brand'] = (merge['brand_name'].notnull()).astype('int32')
        print('Has_brand filled.')
        
        columns = ['name', 'category_name', 'item_description'] # 'brand_name',
        for col in columns: merge[col].fillna(value=MISSVALUE, inplace=True)
        merge['brand_name'].fillna(value='', inplace=True)

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
    y_train = ((np.log1p(dfAll.price) - PRICE_MEAN) / PRICE_STD).values[:TRAIN_SIZE]
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
    def __init__(self, sparse_merge, price, desc,name):
        self.sparse_merge=sparse_merge
        self.desc=desc
        self.name=name
        self.price=price
        self.splitter=self.split_data()

    def split_data(self):
        splitter=[]
        cv = KFold(n_splits=cv_fold, shuffle=False, random_state=2018)
        for train_ids, valid_ids in cv.split(self.price):
            splitter.append([train_ids.copy(), valid_ids.copy()])
        return splitter

    def _get_train_valid_ind(self,i):
        train_ind,test_ind=self.splitter[i]
        y_train=self.price[train_ind]
        y_valid=self.price[test_ind]
        return train_ind,test_ind, y_train, y_valid

class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        if self.learner_name in ["WordBatch",'best_WordBatch']:
            return vanila_WorldBatch_Regressor(param_dict=self.param_dict)

    def predict(self, sparse_merge,desc,name,y_train,train_ind, test_ind):
        y_pred = self.learner.predict(sparse_merge,desc,name,y_train,train_ind, test_ind)
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

        cv_fold=1
        rmse_cv = np.zeros((cv_fold))
        for i in range(cv_fold):
            train_ind, test_ind,y_train, y_valid= self.feature._get_train_valid_ind(i)
            # fit
            y_pred = self.learner.predict(self.feature.sparse_merge,self.feature.desc,self.feature.name,y_train,train_ind, test_ind)
            
            _save('Data/Tfm', [y_pred,y_valid])
            # fit
            rmse_cv[i] = self.compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred.reshape(-1)*PRICE_STD)
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
    def __init__(self, learner_name, sparse_merge, price, desc,name,logger):
        self.learner_name = learner_name
        self.logger = logger
        self.feature = self._get_feature(sparse_merge, price, desc,name)
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self,sparse_merge, price, desc,name):
        feature = Feature(sparse_merge, price, desc,name)
        return feature

    def _obj(self, param_dict):
        self.trial_counter += 1
        print('learner_name: %s, trial_counter: %s'%(self.learner_name,self.trial_counter))
        start_time = time.time()
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict)
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
        
def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This method has not been modified from the original.
    Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]

class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None        

def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()
    
def main():
    feature_vectorized_file_name='Data/feature_vectorized'
    if os.path.exists(feature_vectorized_file_name)==False:
        sparse_merge, price, desc,name=_load(feature_vectorized_file_name)
        print(sparse_merge.shape)
    else:
        ########################################################################
        start_time = time.time()
        merge, submission, price = get_extract_feature()
        merge=merge[:TRAIN_SIZE]

        merge['item_condition_id'] = merge['item_condition_id'].astype('category')
        print('[{}] Convert categorical completed'.format(time.time() - start_time))

       
        brands_filling(merge)
        
        merge['gencat_cond'] = merge['category_1'].map(str) + '_' + merge['item_condition_id'].astype(str)
        merge['subcat_1_cond'] = merge['category_2'].map(str) + '_' + merge['item_condition_id'].astype(str)
        merge['subcat_2_cond'] = merge['category_name'].map(str) + '_' + merge['item_condition_id'].astype(str)
        print(f'[{time.time() - start_time}] Categories and item_condition_id concancenated.')
        
        # vectorize features
        wb = CountVectorizer()
        X_category2 = wb.fit_transform(merge['category_2'])
        X_category3 = wb.fit_transform(merge['category_name'])
        X_brand2 = wb.fit_transform(merge['brand_name'])
        wb = CountVectorizer(token_pattern='.+',min_df=2)
        X_gencat_cond = wb.fit_transform(merge['gencat_cond'])
        X_subcat_1_cond = wb.fit_transform(merge['subcat_1_cond'])
        X_subcat_2_cond = wb.fit_transform(merge['subcat_2_cond'])
        print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

        lb = LabelBinarizer(sparse_output=True)
        X_brand = lb.fit_transform(merge['brand_name'])
        X_category1 = lb.fit_transform(merge['category_1'])
        X_category4 = lb.fit_transform(merge['category_name'])
        print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

        X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values)

        
        # hand feature
        columns = ['brand_name_Frec', 'item_description_wordLen']
        for col in columns:
            merge[col] = np.log1p(merge[col])
            merge[col] = merge[col] / merge[col].max()

        hand_feature = ['brand_name_Frec', 'item_description_wordLen', 'brand_name_name_Intsct',
                        'brand_name_item_description_Intsct','has_brand']

        X_hand_feature = merge[hand_feature].values

        sparse_merge = hstack((X_dummies, X_brand, X_brand2, X_category1, X_category2, X_category3, X_category4, X_hand_feature,
                              X_gencat_cond,X_subcat_1_cond,X_subcat_2_cond)).tocsr()

        print(X_dummies.shape, X_brand.shape, X_brand2.shape, X_category1.shape, X_category2.shape, X_category3.shape, X_category4.shape,
              X_hand_feature.shape,sparse_merge.shape,X_gencat_cond.shape,X_subcat_1_cond.shape,X_subcat_2_cond.shape)

                
        merge['item_description'] = merge['category_2'].map(str)+' . . '+\
                                    merge['name'].map(str)+' . . '+\
                                    merge['item_description'].map(str)
        
        desc=merge['item_description']
        name=merge['name']

        _save(feature_vectorized_file_name, [sparse_merge,price,desc,name])
        print('[{}] data saved.'.format(time.time() - start_time))

    ########################################################################
    # use hyperopt to find the best parameters of the model
    learner_name='best_WordBatch'
    #learner_name='WordBatch'
    print(learner_name)
    logname = "[Learner@%s]_hyperopt_%s.log"%(learner_name,time_utils._timestamp())
    logger = logging_utils._get_logger('Log', logname)
    logger.info('start')

    optimizer=TaskOptimizer(learner_name,sparse_merge, price, desc,name,logger)
    optimizer.run()

if __name__ == "__main__":
    main()

























































