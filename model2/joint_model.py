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
from encoder import create_encoder
from decoder import create_decoder

class JointModel(tf.keras.Model):

    def __init__(self,units,vocab_size,use_att= False,dropout_rate=0.5,isBidirectional=True):

        super(JointModel,self).__init__()

        self._dropout_rate = dropout_rate
        self._isBidirectional = isBidirectional

        self._units = units

        ## encoder进行特征抽取
        self._encoder = create_encoder(self._units,dropout_rate = dropout_rate,isBidirectional=isBidirectional)

        if use_att:
            name = 'ATT'
        else:
            name = 'basic'

        ## decoder进行序列解析
        self._decoder = create_decoder(name,self._units,dropout_rate,False)

        ## classification的decoder
        self._clas_decoder = create_decoder(name,self._units,dropout_rate,False,vocab_size)

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

        classification_result,_ = self._clas_decoder(dec_input,dec_hidden,enc_output,predict=predict)

        return regression_result,classification_result

