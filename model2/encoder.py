#coding:utf-8
'''
定义encoder
'''
from base_layer import gru_layer
import tensorflow as tf

def create_encoder(units,dropout_rate,isBidirectional):
    return Encoder(units,dropout_rate,isBidirectional)

class Encoder(tf.keras.Model):

    def __init__(self,enc_units,dropout_rate = 0.5,isBidirectional=False):
        super(Encoder,self).__init__()
        self._enc_units = enc_units

        self._vector = tf.keras.layers.Dense(self._enc_units,activation='tanh')
        self._vector_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self._gru = gru_layer(self._enc_units,isBidirectional)

        self._isBidirectional = isBidirectional

        self._gru_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    ##定义前向传播方法
    def call(self,x,hidden,predict = False):

        x = self._vector(x)
        if not predict:
            x = self._vector_dropout(x)
        
        ## 使用gru的RNN进行前向传播，得到每一步的输出以及state
        outs = self._gru(x,initial_state = hidden)

        if self._isBidirectional:
            output,fw_state,bw_state = outs
            # print(state.shape)
            state = fw_state+bw_state
        else:
            output,state = outs

        if not predict:
            output = self._gru_dropout(output)

        return output,state

    def initialize_hidden_state(self,size,isBidirectional):
        ## 初始状态是 batch size x enc_units的大小
        if not isBidirectional:
            return tf.zeros((size,self._enc_units),tf.float64)
        else:
            return [tf.zeros((size,self._enc_units),tf.float64),tf.zeros((size,self._enc_units),tf.float64)]