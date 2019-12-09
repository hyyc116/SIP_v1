#coding:utf-8
'''
jiont model

'''
import sys
sys.path.extend(['..','.'])
from paths import PATH
from basic_config import *
from tensorflow.keras import layers
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from encoder import Encoder
from decoder import Decoder

class JointModel(tf.keras.Model):

    def __init__(self,units,vocab_size,use_att= False,dropout_rate=0.5,isBidirectional=True):

        super(JointModel,self).__init__()

        self._dropout_rate = dropout_rate
        self._isBidirectional = isBidirectional

        self._units = units

        ## encoder进行特征抽取
        self._encoder = create_encoder(self._units,dropout_rate = dropout_rate,isBidirectional=isBidirectional)

        if use_att:

            name = 'att_decoder'

        else:
            name = 'basic'

        ## decoder进行序列解析
        self._decoder = create_decoder(name,self._units,dropout_rate)

        ## 加一层中间层
        self._fc1 = tf.keras.layers.Dense(self._units,activation='sigmoid')
        self._drp2 = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 进行label的预测
        self._fc = tf.keras.layers.Dense(vocab_size)


    def call(self,X,Y,L,predict=False):

        batch_size = X.shape[0]

        ## 首先需要encoder对数据进行特征抽取
        initial_state = self._encoder.initialize_hidden_state(batch_size,self._isBidirectional)
        ## 首先对输入进行encode
        enc_output, enc_hidden = self._encoder(X,initial_state,predict=predict)

        ## regression decoder
        dec_input = tf.cast(tf.expand_dims([0]*batch_size,1),tf.float64)
        dec_hidden = enc_hidden
        all_predictions = []
        for t in range(Y.shape[1]):

            predictions,dec_hidden = self._decoder(dec_input,dec_hidden,enc_output,predict=predict)

            all_predictions.append(predictions)
            dec_input = predictions

        regression_result = tf.concat(all_predictions,1)

        ## classification
        enc_output = tf.reshape(enc_output,(enc_output.shape[0],-1))
        # print(enc_output.shape)
        classification_result = self._fc(enc_output)

        return regression_result,classification_result

