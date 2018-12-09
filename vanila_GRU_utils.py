# -*- coding: utf-8 -*-
# @Time    : 1/18/18 9:08 AM
# @Author  : LeeYun
# @File    : vanila_GRU_utils2.py.py
'''Description :

'''
import time,os,math
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras import backend,callbacks
from keras.backend import one_hot
from keras import optimizers
from keras.layers import Lambda
from keras.layers import Bidirectional,GlobalMaxPool1D
from config import PRICE_STD

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_loss(pred, y):
    return np.sqrt(np.mean(np.square(pred - y)))

class Middle_Predict(callbacks.Callback):
    def __init__(self,X_valid,valid_len,batch_size,train_len,weight):
        super().__init__()
        self.X_valid=X_valid
        self.batch_size=batch_size
        self.pred=np.zeros(valid_len)
        self.epoch=0
        self.weight=weight
        self.point=0

        total_batches=math.ceil(train_len/batch_size)
        self.pred_epoch = []
        for i in range(3):
            self.pred_epoch.append(total_batches - 1 - 150 * i)

    def on_batch_end(self, batch, logs={}):
        if self.epoch==1 and (batch in self.pred_epoch):
            print(batch)
            self.pred=self.pred+self.model.predict(self.X_valid, batch_size=self.batch_size).reshape(-1)*self.weight[self.point]
            self.point+=1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch+=1

    def on_train_end(self, logs={}):
        self.pred/=self.pred_epoch.__len__()

class vanila_GRU_Regressor:
    def __init__(self, param_dict,Item_size):
        weight1=param_dict['weight1']
        weight2=param_dict['weight2']
        weight3=param_dict['weight3']
        total=weight1+weight2+weight3
        self.weight=[weight1/total*3,weight2/total*3,weight3/total*3]

        self.batch_size=param_dict['batch_size']
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

        init = tf.global_variables_initializer()
        self.session = backend.get_session()
        self.init = init

    def fit(self, X_train, y_train, X_valid, y_valid, train_len, valid_len):
        self.session.run(self.init)
        t0 = time.time()
        ensemble_predict=Middle_Predict(X_valid,valid_len,self.batch_size,train_len,self.weight)
        self.model.fit(X_train, y_train, epochs=2, batch_size=self.batch_size, verbose=2, callbacks=[ensemble_predict])
        took = time.time() - t0
        print('took %.3fs' % (took))

        return ensemble_predict.pred
        # loss = compute_loss(y_valid.reshape(-1) * PRICE_STD, ensemble_predict.pred * PRICE_STD)
        # print('final loss: %f' % (loss))

    def predict(self,X_valid,valid_len):
        preds = self.model.predict(X_valid,batch_size=self.batch_size).reshape(-1)
        return preds


    def close_session(self):
        backend.clear_session()













































