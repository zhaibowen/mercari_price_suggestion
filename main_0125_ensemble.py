# -*- coding: utf-8 -*-
# @Time    : 1/15/18 9:03 AM
# @Author  : LeeYun
# @File    : main_0111_gru.py
'''Description :

'''
import os,numba,string,pickle,regex,nltk,time,time_utils,logging_utils
from vanila_ensemble_utils import vanila_ensemble_Regressor, pretrain_model
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

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

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
    def __init__(self):
        self.X_valid, self.y_valid= pretrain_model()

    def _get_valid_data(self,i):
        return self.X_valid[i], self.y_valid[i]

class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        if self.learner_name in ["vanila_ensemble",'best_vanila_ensemble']:
            return vanila_ensemble_Regressor(param_dict=self.param_dict)

    def predict(self, X_valid):
        # X_valid[-1,3]  y_pred[-1]
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

        rmse_cv = np.zeros((cv_fold))
        for i in range(cv_fold):
            X_valid, y_valid = self.feature._get_valid_data(i)
            y_pred = self.learner.predict(X_valid)
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
    def __init__(self, learner_name, logger):
        self.learner_name = learner_name
        self.logger = logger
        self.feature = self._get_feature()
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self):
        feature = Feature()
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

def main():
    ########################################################################
    # use hyperopt to find the best parameters of the model
    # use 5 fold cross validation
    learner_name='vanila_ensemble'
    print(learner_name)
    logname = "[Learner@%s]_hyperopt_%s.log"%(learner_name,time_utils._timestamp())
    logger = logging_utils._get_logger('Log', logname)
    logger.info('start')

    optimizer=TaskOptimizer(learner_name,logger)
    optimizer.run()

if __name__ == "__main__":
    main()

























































