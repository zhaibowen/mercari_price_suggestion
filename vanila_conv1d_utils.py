# -*- coding: utf-8 -*-
# @Time    : 1/18/18 9:08 AM
# @Author  : LeeYun
# @File    : vanila_GRU_utils2.py.py
'''Description :

'''
import time,pickle
import numpy as np
import tensorflow as tf
from config import PRICE_STD

def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)
    
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

def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res

def compute_loss(pred, y):
    return np.sqrt(np.mean(np.square(pred - y)))

class vanila_conv1d_Regressor:
    def __init__(self, param_dict,Item_size):
        weight1=param_dict['weight1']
        weight2=param_dict['weight2']
        weight3=param_dict['weight3']
        total=weight1+weight2+weight3
        self.weight=[weight1/total*3,weight2/total*3,weight3/total*3]

        self.seed=2018
        self.batch_size=param_dict['batch_size']
        self.lr=param_dict['lr']

        name_seq_len=param_dict['name_Len']
        desc_seq_len=param_dict['description_Len']
        denselayer_units=param_dict['denselayer_units']
        embed_name=param_dict['embed_name']
        embed_desc=param_dict['embed_desc']
        embed_brand=param_dict['embed_brand']
        embed_cat_2=param_dict['embed_cat_2']
        embed_cat_3=param_dict['embed_cat_3']
        name_filter=param_dict['name_filter']
        desc_filter=param_dict['desc_filter']
        name_filter_size=param_dict['name_filter_size']
        desc_filter_size=param_dict['desc_filter_size']
        dense_drop=param_dict['dense_drop']

        name_voc_size=Item_size['name']
        desc_voc_size=Item_size['item_description']
        brand_voc_size=Item_size['brand_name']
        cat1_voc_size=Item_size['category_1']
        cat2_voc_size=Item_size['category_2']
        cat3_voc_size=Item_size['category_name']

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = self.seed
        with graph.device("/gpu:1"):
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

                self.is_train=tf.placeholder(tf.bool, shape=(), )

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

                out = tf.concat([name, desc, brand, cat_1, cat_2, cat_3, ship, cond, hand_feat], axis=1) #
                
                # print(name.shape)
                # print(desc.shape)
                # print(brand.shape)
                # print(cat_1.shape)
                # print(cat_2.shape)
                # print(cat_3.shape)
                # print(ship.shape)
                # print(cond.shape)
                # print(hand_feat.shape)
                # print('concatenated dim:', out.shape)

                out = dense(out, denselayer_units, activation=tf.nn.relu)
                out = tf.layers.dropout(out, rate=dense_drop, training=self.is_train)
                self.out = dense(out, 1)

                loss = tf.losses.mean_squared_error(self.place_y, self.out)
                opt = tf.train.AdamOptimizer(learning_rate=self.place_lr)
                self.train_step = opt.minimize(loss)

                init = tf.global_variables_initializer()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # self.session = tf.Session(config=config)
            self.session = tf.Session(config=config,graph=graph)
            self.init=init

    def fit(self, X_train, y_train, X_valid, y_valid, train_len, valid_len):
        self.session.run(self.init)

        test_idx = np.arange(valid_len)
        test_batches = prepare_batches(test_idx, self.batch_size)
        y_pred = np.zeros(valid_len)
            
        EPOCH=3
        
        for epoch in range(EPOCH):
            t0 = time.time()
            np.random.seed(epoch)
            

            train_idx_shuffle = np.arange(train_len)
            np.random.shuffle(train_idx_shuffle)
            batches = prepare_batches(train_idx_shuffle, self.batch_size)

            pred = []
            batch_len=batches.__len__()
            pred_epoch=[]
            for i in range(3):
                pred_epoch.append(batch_len-1-200*i)

            for rnd,idx in enumerate(batches):
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

                # if rnd%50==0:
                if epoch==EPOCH-1 and (rnd in pred_epoch):
                    for idx in test_batches:
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
                    # loss = compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
                    # print('epoch: %d, round: %d, loss: %f'%(epoch,rnd,loss))
                    pred.append(y_pred.copy())

            # took = time.time() - t0
            # print('epoch %d took %.3fs' % (epoch, took))

        y_pred = np.zeros(valid_len)
        for item,weight in zip(pred,self.weight):
            y_pred=y_pred+item*weight
        y_pred/=3
        # loss = compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
        # print('final loss: %f'%(loss))
        #_save('Data/Tfasttext2', [y_pred,y_valid])
        
        return y_pred

    def predict(self,X_valid,valid_len):
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

        return y_pred

    def close_session(self):
        self.session.close()











































