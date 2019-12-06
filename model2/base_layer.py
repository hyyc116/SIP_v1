#coding:utf-8
'''
定义一些基本的layer
'''

import tensorflow as tf

def gru(units,go_backwards=False):
    return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform',
                                   go_backwards=go_backwards)

def biGru(units):

    fw_gru = gru(units)
    # bw_gru = gru(units,go_backwards=True)

    return tf.keras.layers.Bidirectional(fw_gru)


def gru_layer(units,isBidirectional=False):

    if isBidirectional:
        return biGru(units)

    else:
        return gru(units)

