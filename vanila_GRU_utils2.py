# -*- coding: utf-8 -*-
# @Time    : 1/18/18 9:08 AM
# @Author  : LeeYun
# @File    : vanila_GRU_utils2.py.py
'''Description :

'''
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,GRUBlockCellV2
from config import PRICE_STD

def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup

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

def make_encoder(sequence, output_dim, seed):
    rnn_cell=GRUCell(num_units=output_dim,
                kernel_initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05,dtype=tf.float32, seed=seed),
                bias_initializer=tf.zeros_initializer())
    rnn_out, rnn_state  = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=tf.transpose(sequence, [1, 0, 2]),
        initial_state=rnn_cell.zero_state(tf.shape(sequence)[0], dtype=tf.float32),
        time_major=True)
    return rnn_state

def compute_loss(pred, y):
    return np.sqrt(np.mean(np.square(pred - y)))

class vanila_GRU_Regressor:
    def __init__(self, param_dict,Item_size):
        seed=2018
        self.batch_size=param_dict['batch_size']
        self.lr=param_dict['lr']
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

        tf.reset_default_graph()
        graph = tf.Graph()
        graph.seed = seed
        with graph.device("/gpu:0"):
            with graph.as_default():
                self.place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
                self.place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
                self.place_brand = tf.placeholder(tf.int32, shape=(None, 1))
                self.place_cat1 = tf.placeholder(tf.int32, shape=(None, 1))
                self.place_cat2 = tf.placeholder(tf.int32, shape=(None, 1))
                self.place_cat3 = tf.placeholder(tf.int32, shape=(None, 1))
                self.place_ship = tf.placeholder(tf.float32, shape=(None, 1))
                self.place_cond = tf.placeholder(tf.uint8, shape=(None, 1))
                self.place_hand = tf.placeholder(tf.float32, shape=(None, hand_feature_cols))

                self.place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

                self.place_lr = tf.placeholder(tf.float32, shape=(), )

                self.is_train = tf.placeholder(tf.bool, shape=(), )

                name = embed(self.place_name, name_voc_size, embed_name)
                desc = embed(self.place_desc, desc_voc_size, embed_desc)
                brand = embed(self.place_brand, brand_voc_size, embed_brand)
                cat_2 = embed(self.place_cat2, cat2_voc_size, embed_cat_2)
                cat_3 = embed(self.place_cat3, cat3_voc_size, embed_cat_3)

                with tf.variable_scope("h_name"):
                    name = make_encoder(name, output_dim=rnn_dim_name, seed=seed)
                with tf.variable_scope("h_desc"):
                    desc = make_encoder(desc, output_dim=rnn_dim_desc, seed=seed)

                brand = tf.contrib.layers.flatten(brand)

                cond = tf.one_hot(self.place_cond, 5)
                cond = tf.contrib.layers.flatten(cond)

                cat_1 = tf.one_hot(self.place_cat1, cat1_voc_size)
                cat_1 = tf.contrib.layers.flatten(cat_1)

                cat_2 = tf.contrib.layers.flatten(cat_2)
                cat_3 = tf.contrib.layers.flatten(cat_3)

                hand_feat = self.place_hand
                ship = self.place_ship

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

            config = tf.ConfigProto(allow_soft_placement = True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config,graph=graph)
            self.init=init

    def fit(self, X_train, y_train, X_valid, y_valid, train_len, valid_len):
        self.session.run(self.init)
        for epoch in range(2):
            t0 = time.time()
            np.random.seed(epoch)

            train_idx_shuffle = np.arange(train_len)
            np.random.shuffle(train_idx_shuffle)
            batches = prepare_batches(train_idx_shuffle, self.batch_size)

            test_idx = np.arange(valid_len)
            test_batches = prepare_batches(test_idx, self.batch_size)
            y_pred = np.zeros(valid_len)

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

                if rnd%1000==0:
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
                    loss = compute_loss(y_valid.reshape(-1)*PRICE_STD, y_pred*PRICE_STD)
                    print('epoch: %d, round: %d, loss: %f'%(epoch,rnd,loss))

            took = time.time() - t0
            print('epoch %d took %.3fs' % (epoch, took))

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













































