#coding:utf-8
'''
定义各种attention

'''
import tensorflow as tf
class BahdanauAttention(tf.keras.Model):

    def __init__(self,units):
        super(BahdanauAttention,self).__init__()

        self._W1 = tf.keras.layers.Dense(units)

        self._W2 = tf.keras.layers.Dense(units)

        self._V = tf.keras.layers.Dense(1)

    def call(self,query,values):

        hidden_with_time_axis = tf.expand_dims(query,1)

        score = self._V(tf.nn.tanh(self._W1(values)+self._W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score,axis=1)

        context_vector = attention_weights*values

        context_vector = tf.reduce_sum(context_vector,axis=1)

        return context_vector,attention_weights