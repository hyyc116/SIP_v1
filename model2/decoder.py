#coding:utf-8
'''
定义Decoder

'''
import tensorflow as tf
from attentions import BahdanauAttention
from base_layer import gru_layer

def create_decoder(name,units,dropout_rate):

    if name=='basic':
        return BasicDecoder(units,dropout_rate)

    elif name=='att_decoder':
        return AttDecoder(units,dropout_rate)

    else:
        print('create basic decoder as default.')
        return BasicDecoder(units,dropout_rate)


class AttDecoder(tf.keras.Model):

    def __init__(self,dec_units,dropout_rate=0.5):

        super(AttDecoder,self).__init__()

        self._dec_units = dec_units

        ## attention
        self._attention = BahdanauAttention(self._dec_units)

        ## dense用于对连续值embedding
        self._emd_fc = tf.keras.layers.Dense(self._dec_units,activation='tanh')

        ## GRU
        self._gru = gru_layer(self._dec_units)
        self._rnn_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 回归 每一步输出一个数字
        self._fc = tf.keras.layers.Dense(1)


    def call(self,dec_input,dec_hidden,enc_output,predict=False):

        ## 对decoder的输入进行embedding
        dec_input = self._emd_fc(dec_input)
        ## 使用hidden state以及enc_outputs计算attention
        context_vector,weights = self._attention(dec_hidden,enc_output)

        x = tf.concat([context_vector,dec_input],axis=-1)

        if len(x.shape)==2:
            x = tf.expand_dims(x,1)

        output,state = self._gru(x)

        if not predict:
            output = self._rnn_dropout(output)

        output = tf.reshape(output,(dec_input.shape[0],-1))
        output = self._fc(output)

        return output,state

    def initialize_hidden_state(self,size):
        return tf.zeros((size,self._dec_units),tf.float64)


class BasicDecoder(tf.keras.Model):

    def __init__(self,dec_units,dropout_rate=0.5):

        super(BasicDecoder,self).__init__()

        self._dec_units = dec_units

        ## dense用于对连续值embedding
        self._emd_fc = tf.keras.layers.Dense(self._dec_units,activation='tanh')

        self._gru = gru(self._dec_units)

        self._rnn_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 回归 每一步输出一个数字
        self._fc = tf.keras.layers.Dense(1)


    def call(self,dec_input,dec_hidden,enc_output,predict=False):

        ## 基础的Decoder,将上一步的结果与上一步的hidden state传入
        dec_input = self._emd_fc(dec_input)
        x = tf.concat([dec_input,dec_hidden],axis=-1)

        ## 在实际运用过程中每次deocde只进行一步
        if len(x.shape)==2:
            x = tf.expand_dims(x,1)

        output,state = self._gru(x)

        if not predict:
            output = self._rnn_dropout(output)

        ## batch size x 1
        x = tf.reshape((self._fc(output)),(x.shape[0],-1))

        return x,state

    def initialize_hidden_state(self,size):
        return tf.zeros((size,self._dec_units),tf.float64)


