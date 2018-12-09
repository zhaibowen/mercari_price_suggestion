# -*- coding: utf-8 -*-
# @Time    : 1/15/18 9:08 PM
# @Author  : LeeYun
# @File    : config.py
'''Description :

'''
from hyperopt import hp
import numpy as np

TRAIN_SIZE = 1481661
TEST_SIZE = 693359
PRICE_MEAN = 2.98081628517
PRICE_STD = 0.7459273872548303
NumWords = 50000
MISSVALUE='missvalue'

hp_iter = 1000
cv_fold = 5

param_space_GRU_ensemble = {
    'weight1': hp.uniform("weight1", 0, 1),
    'weight2': hp.uniform("weight2", 0, 1),
    'weight3': hp.uniform("weight3", 0, 1),
}

param_space_best_GRU_ensemble = {
    'weight1': 0.4,
    'weight2': 0.5,
    'weight3': 0.6,
}

param_space_conv1d_ensemble = {
    'weight1': hp.uniform("weight1", 0, 1),
    'weight2': hp.uniform("weight2", 0, 1),
    'weight3': hp.uniform("weight3", 0, 1),
}

param_space_best_conv1d_ensemble = {
    'weight1': 0.8,
    'weight2': 1.0,
    'weight3': 1.2,
}

param_space_ensemble = {
    'GRU_weight': hp.uniform("GRU_weight", 0, 1),
    'conv1d_weight': hp.uniform("conv1d_weight", 0, 1),
    'FM_FTRL_weight': hp.uniform("FM_FTRL_weight", 0, 1),
}

param_space_best_ensemble = {
    'GRU_weight': 0.5332235012111342,
    'conv1d_weight': 0.5272285958816297,
    'FM_FTRL_weight': 0.9188890188824305,
}

param_space_vanila_GRU = {
    'denselayer_units': hp.uniform("denselayer_units", 128, 256),
    'description_Len': hp.uniform("description_Len", 80, 90),
    'name_Len': hp.uniform("name_Len", 8, 12),
    'embed_name': hp.uniform("embed_name", 45, 61),
    'embed_desc': hp.uniform("embed_desc", 45, 61),
    'embed_brand': hp.uniform("embed_brand", 20, 31),
    'embed_cat_2': hp.uniform("embed_cat_2", 8, 16),
    'embed_cat_3': hp.uniform("embed_cat_3", 30, 41),
    'rnn_dim_name': hp.uniform("rnn_dim_name",20, 30),
    'rnn_dim_desc': hp.uniform("rnn_dim_desc",20, 30),
    'lr': hp.loguniform("lr", np.log(0.001), np.log(0.1)),
    'batch_size': hp.uniform("batch_size",512, 2048),
    'dense_drop': hp.loguniform("dense_drop", np.log(0.001), np.log(0.01)),
}

# 0.42395  959s
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

# param_space_vanila_con1d = {
#     'denselayer_units': hp.uniform("denselayer_units", 128, 256),
#     'description_Len': 85,
#     'name_Len': 10,
#     'embed_name': hp.uniform("embed_name", 45, 61),
#     'embed_desc': hp.uniform("embed_desc", 45, 61),
#     'embed_brand': hp.uniform("embed_brand", 20, 31),
#     'embed_cat_2': hp.uniform("embed_cat_2", 8, 16),
#     'embed_cat_3': hp.uniform("embed_cat_3", 30, 41),
#     'name_filter': hp.uniform("name_filter",100, 150),
#     'desc_filter': hp.uniform("desc_filter",100, 150),
#     'name_filter_size': 4,
#     'desc_filter_size': 4,
#     'lr': hp.loguniform("lr", np.log(0.001), np.log(0.01)),
#     'batch_size': hp.uniform("batch_size",512, 2048),
#     'dense_drop': hp.loguniform("dense_drop", np.log(0.001), np.log(0.01)),
#     'weight1': hp.uniform("weight1",0, 1),
#     'weight2': hp.uniform("weight2",0, 1),
#     'weight3': hp.uniform("weight3",0, 1),
# }
param_space_vanila_con1d = {
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
    'lr': hp.loguniform("lr", np.log(0.0008), np.log(0.02)),
    'batch_size': hp.uniform("batch_size",700, 1200),
    'dense_drop': 0.009082372998548981,
    'weight1': hp.uniform("weight1",0, 1),
    'weight2': hp.uniform("weight2",0, 1),
    'weight3': hp.uniform("weight3",0, 1),
}

# 0.42192  621s
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
    'name_filter_size': 3,
    'desc_filter_size': 3,
    'lr': 0.003924230325700921,
    'batch_size': 933,
    'dense_drop': 0.009082372998548981,
    'weight1': 0.8,
    'weight2': 1.0,
    'weight3': 1.2,
}

# 0.41422  embend*4
# fasttest={
#     'batch_size': 1177,
#     'lr': 0.004565707978962733,
#     weight=[
#         0.9116810454224276,
#         0.9263363989954573,
#         1.1619825555821153,
#     ]
# }

param_space_FM_FTRL={
    'alpha': hp.uniform("alpha", 0.01, 0.033),
    'beta': hp.uniform("beta", 0.0005, 0.0015),
    'L1': hp.uniform("L1", 1.6e-05, 3.6e-05),
    'L2': hp.uniform("L2", 0.005, 0.02),
    'alpha_fm': hp.uniform("alpha_fm", 0.006, 0.02),
    'init_fm':hp.uniform("init_fm", 0.015, 0.055),
    'D_fm': hp.uniform("D_fm",240, 255),
    'e_noise': hp.uniform("e_noise", 0.0003, 0.00085),
    'iters': 8,
}

# 0.41217
param_space_best_FM_FTRL = {
    'alpha': 0.027337603769727097,
    'beta': 0.000696315247353981,
    'L1': 3.261127986664043e-05,
    'L2': 0.01154650635257946,
    'alpha_fm': 0.01551798270107723,
    'init_fm': 0.02798274281370472,
    'D_fm': 248,
    'e_noise': 0.0007545501112447097,
    'iters': 9,
}

# 41041
# param_space_best_FM_FTRL = {
#     'alpha': 0.03273793453882604,
#     'beta': 0.0011705530094547533,
#     'L1': 3.59507400149913e-05,
#     'L2': 0.018493058691917252,
#     'alpha_fm': 0.015903973928100217,
#     'init_fm': 0.02883106077640207,
#     'D_fm': 247,
#     'e_noise': 0.0003029146164926251,
#     'iters': 8,
# }

# param_space_best_FM_FTRL2 = {
#     'alpha': 0.0328056810267008,
#     'beta': 0.000816193943215732,
#     'L1': 3.305523172523853e-05,
#     'L2': 0.018582167239926114,
#     'alpha_fm': 0.018278643423563678,
#     'init_fm': 0.029241816326542665,
#     'D_fm': 249,
#     'e_noise': 0.0004526632152484058,
#     'iters': 6,
# }
        
param_space_FTRL={
    'alpha': hp.loguniform("alpha", np.log(0.001), np.log(0.1)),
    'beta': hp.loguniform("beta", np.log(0.01), np.log(1)),
    'L1': hp.loguniform("L1", np.log(0.000001), np.log(0.0001)),
    'L2': hp.loguniform("L2", np.log(0.01), np.log(5)),
    'iters': hp.uniform("iters",10, 30),
}

# 0.43296
param_space_best_FTRL={
    'alpha': 0.035791106596213554,
    'beta': 0.04670014156515456,
    'L1': 1.4562387067658935e-06,
    'L2': 0.26754419523908257,
    'iters': 17,
}

param_space_WordBatch={
    'desc_w1': hp.uniform("desc_w1", 1.0, 1.5),
    'desc_w2': hp.uniform("desc_w2", 1.0, 1.5),
    'name_w1': hp.uniform("name_w1", 1.8, 2.5),
    'name_w2': hp.uniform("name_w2", 0.2, 0.7),
    'alpha': hp.uniform("alpha", 0.01, 0.033),
    'beta': hp.uniform("beta", 0.0005, 0.0015),
    'L1': hp.uniform("L1", 1.6e-05, 3.6e-05),
    'L2': hp.uniform("L2", 0.005, 0.02),
    'alpha_fm': hp.uniform("alpha_fm", 0.006, 0.02),
    'init_fm':hp.uniform("init_fm", 0.015, 0.055),
    'D_fm': hp.uniform("D_fm",245, 256),
    'e_noise': hp.uniform("e_noise", 0.0003, 0.00085),
    'iters': 10,
}


param_space_best_WordBatch={
    'desc_w1': 1.3740067995315037,
    'desc_w2': 1.0248685266832964,
    'name_w1': 2.1385527373939834,
    'name_w2': 0.3894761681383836,
}

param_space_dict = {
    "GRU_ensemble": param_space_GRU_ensemble,
    "best_GRU_ensemble": param_space_best_GRU_ensemble,
    "con1d_ensemble": param_space_conv1d_ensemble,
    "best_con1d_ensemble": param_space_best_conv1d_ensemble,
    "vanila_ensemble": param_space_ensemble,
    "best_vanila_ensemble": param_space_best_ensemble,
    "vanila_GRU": param_space_vanila_GRU,
    "best_vanila_GRU": param_space_best_vanila_GRU,
    "vanila_con1d": param_space_vanila_con1d,
    'best_vanila_con1d': param_space_best_vanila_con1d,
    'FM_FTRL': param_space_FM_FTRL,
    'best_FM_FTRL': param_space_best_FM_FTRL,
    'FTRL': param_space_FTRL,
    'best_FTRL': param_space_best_FTRL,
    'WordBatch': param_space_WordBatch,
    'best_WordBatch': param_space_best_WordBatch,
}

threshold_dict = {
    "vanila_GRU": 0.425,
    "best_vanila_GRU": 0.425,
    "vanila_con1d": 0.424,
    'best_vanila_con1d': 0.424,
}

int_params = [
    'denselayers','description_Len','name_Len','D_fm','iters','max_iter','batch_size',
    'denselayer_units','embed_name','embed_desc','embed_brand','embed_cat_2','embed_cat_3',
    'name_filter','desc_filter','name_filter_size','desc_filter_size','rnn_dim_name','rnn_dim_desc'
]











































