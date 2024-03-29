#coding:utf-8
'''
本文件使用LSTM进行预测

'''
import sys
sys.path.extend(['..','.'])
from paths import PATH
from basic_config import *
from tensorflow.keras import layers
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
import time

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

tf.keras.backend.set_floatx('float64')
from datasets import construct_RNN_datasets
from datasets import unscale_dataset
from base_layer import gru
from encoder import create_encoder
from decoder import create_decoder

from losses import regress_mse_loss
from losses import regress_mae_loss
from losses import regress_huber_loss

from tools import is_better_result


class S2SM:

    def __init__(self,pathObj,m,n,scale = True,feature_set='basic',use_att=True,seperate_static=False,isBidirectional=False,use_l2 = True):

        ## 文件的路径
        self._pathObj = pathObj
        self._m = m
        self._n = n

        ## 是否对数据进行归一化 
        self._scale = scale
        ## 特征集合 basic-author-structure三类
        self._feature_set = feature_set

        ##使用attention
        self._use_att = use_att

        ## 静态变量是否直接进行串联
        self._seperate_static = seperate_static

        ## 是否使用bi-directional的GRU
        self._isBidirectional = isBidirectional

        ## 是否使用l2的weight
        self._use_l2 = use_l2
        if self._use_l2:
            self._l2_weight = 0.001

        ## 加载数据
        if not self._seperate_static:
            self._train_X,self._test_X,self._valid_X,self._dx_mean,self._dx_std,\
            self._train_Y,self._test_Y,self._valid_Y,self._y_mean,self._y_std,\
            self._test_sorted_ids = construct_RNN_datasets(pathObj,m,n,self._scale,self._feature_set)
        else:
            self._train_X,self._test_X,self._valid_X,self._dx_mean,self._dx_std,\
            self._train_SX,self._test_SX,self._valid_SX,self._sx_mean,self._sx_std,\
            self._train_Y,self._test_Y,self._valid_Y,self._y_mean,self._y_std,\
            self._test_sorted_ids = construct_RNN_datasets(pathObj,m,n,self._scale,self._feature_set,seperate_static=self._seperate_static)

        ## 数据集
        self._batch_sz = 512
        self._buffer_size = len(self._train_Y)
        self._n_batchs =self._buffer_size//self._batch_sz

        if not self._seperate_static:
            self._dataset = tf.data.Dataset.from_tensor_slices((self._train_X,self._train_Y)).shuffle(self._buffer_size)
        else:
            self._dataset = tf.data.Dataset.from_tensor_slices((self._train_X,self._train_SX,self._train_Y)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        ### test
        self._test_buffer_size = len(self._test_Y)
        self._n_test_batchs = self._test_buffer_size//self._batch_sz if self._test_buffer_size%self._batch_sz==0 else self._test_buffer_size//self._batch_sz+1

        if not self._seperate_static:
            self._test_dataset = tf.data.Dataset.from_tensor_slices((self._test_X,self._test_Y)).shuffle(self._buffer_size)
        else:
            self._test_dataset = tf.data.Dataset.from_tensor_slices((self._test_X,self._test_SX,self._test_Y)).shuffle(self._test_buffer_size)
        self._test_dataset = self._test_dataset.batch(self._batch_sz, drop_remainder=True)


        self._valid_buffer_size = len(self._valid_Y)
        self._n_valid_batchs = self._valid_buffer_size//self._batch_sz if self._valid_buffer_size%self._batch_sz==0 else self._valid_buffer_size//self._batch_sz+1
        
        if not self._seperate_static:
            self._valid_dataset = tf.data.Dataset.from_tensor_slices((self._valid_X,self._valid_Y)).shuffle(self._buffer_size)
        else:
            self._valid_dataset = tf.data.Dataset.from_tensor_slices((self._valid_X,self._valid_SX,self._valid_Y)).shuffle(self._valid_buffer_size)
        self._valid_dataset = self._valid_dataset.batch(self._batch_sz, drop_remainder=True)

        ## dropout rate
        self._dropout_rate = 0.5
        self._units = 128

        ## 初始化encoder以及decoder
        self._model_name = self.gen_model_name()

        self._encoder = create_encoder(self._units,self._dropout_rate,self._isBidirectional)
        self._decoder = create_decoder(self._model_name,self._units,self._dropout_rate,self._seperate_static)

        print('train model  {}.'.format(self._model_name))

        ## optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4,beta_2=0.95)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)
        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")

        self._trackables = {}
        self._trackables['optimizer']=self._optimizer
        self._trackables['encoder']=self._encoder
        self._trackables['decoder']=self._decoder
        self._checkpoint = tf.train.Checkpoint(**self._trackables)

    def batch_dataset(self,data_tuple,buffer_size,batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(data_tuple).shuffle(buffer_size)
        dataset = dataset.batch(batch_sz, drop_remainder=True)
        return dataset


    def gen_model_name(self):
        self._model_name = 'SIP-m{}-n{}'.format(self._m,self._n)

        self._model_name += '_F-{}'.format(self._feature_set)

        if self._isBidirectional:
            self._model_name+='_Bidirect'
        else:
            self._model_name+='_Signle'

        if self._use_att:
            self._model_name += '_ATT'
        else:
            self._model_name += '_BED'

        if self._seperate_static:
            self._model_name += '_SEP'
        else:
            self._model_name += '_SEQ'


        if self._use_l2:
            self._model_name+='_L2'
        else:
            self._model_name+='_NO'

        return self._model_name

        

    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

    def train_step(self,X,targ,enc_hidden,sx=None):

        with tf.GradientTape() as tape:
            enc_output,enc_hidden = self._encoder(X,enc_hidden)
            dec_input = tf.cast(tf.expand_dims([0]*self._batch_sz,1),tf.float64)

            loss = 0
            dec_hidden = enc_hidden

            all_predictions = []
            for t in range(0,targ.shape[1]):
                predictions,dec_hidden = self._decoder(dec_input,dec_hidden,enc_output,sx=sx)
                loss += regress_mse_loss(tf.expand_dims(targ[:,t],1),predictions)

                all_predictions.append(predictions)

                dec_input = predictions

            all_predictions = tf.concat(all_predictions,axis=1)

            targ = unscale_dataset(targ,self._y_mean,self._y_std)

            all_predictions = unscale_dataset(all_predictions,self._y_mean,self._y_std)

            batch_loss = regress_mse_loss(targ,all_predictions)

            variables = self._encoder.trainable_variables + self._decoder.trainable_variables

            if self._use_l2:

                loss+=tf.reduce_mean([tf.nn.l2_loss(var) for var in variables])*self._l2_weight

            gradients =  tape.gradient(loss,variables)

            self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        early_stop_count = 0
        # best_score = 0
        # best_r2 = 0
        # best_mae = 100
        best_mse = 1000

        test_result = {}
        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self._encoder.initialize_hidden_state(self._batch_sz,self._isBidirectional)

            total_loss = 0

            for (batch,data) in enumerate(self._dataset.take(self._n_batchs)):

                if self._seperate_static:

                    X,SX,targ = data
                    batch_loss  = self.train_step(X,targ,enc_hidden,sx = SX)
                else:
                    X,targ = data
                    batch_loss  = self.train_step(X,targ,enc_hidden)
                
                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:
                    print('Model {}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._model_name,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse = self.batch_predict(self._valid_dataset,self._n_valid_batchs)
            
            logging.info('Model {}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{}.'.format(self._model_name,epoch+1,total_loss,mae,mse,r2))

            if is_better_result(mse,best_mse):

                # best_mae = mae if mae<best_mae else best_mae
                best_mse = mse if mse<best_mse else best_mse
                # best_r2 = r2 if r2>best_r2 else best_r2

                # best_score = r2/(mae+mse)

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                ## 在验证的时候 需要对数据进行unscale
                r2,mae,mse = self.batch_predict(self._test_dataset,self._n_test_batchs)

                logging.info('Model {}, saved model, TEST MAE:{}, MSE:{},R2:{}.'.format(self._model_name,mae,mse,r2))

                test_result['summary'] = 'sip-m{}n{},{},{},{},{}'.format(self._m,self._n,self._model_name,r2,mae,mse)

                early_stop_count=0

            else:
                early_stop_count+=1

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

            ## 如果连续10个回合验证集的loss都不能大于目前保存最好的模型，停止训练
            if early_stop_count>=10:
                break

        ## summary保存到文件里面
        with open(self._pathObj._deep_result_summary,'a') as f:
            summary = test_result['summary']
            f.write(summary+'\n')
            logging.info('Final performace on test is {}.'.format(summary))


    def save_losses(self,loss_obj,model):
        self._pathObj.save_json(self._pathObj.losses_file(self._m,self._n,model),loss_obj)


    def batch_predict(self,dataset,n_batchs):

        print('Predicting, batch size: {} ...'.format(n_batchs))
        mae,mse,r2 = 0.0,0.0,0.0
        for (batch,data) in enumerate(dataset.take(n_batchs)):

            if not self._seperate_static:
                X,Y = data
                SX = None
            else:
                X,SX,Y = data

            valid_size = X.shape[0]
            # print(valid_size,X.shape,Y.shape,self._valid_Y.shape,self._valid_X.shape)
             # 初始化encoder的hideen state
            initial_state = self._encoder.initialize_hidden_state(valid_size,self._isBidirectional)

            ## 输入encoder
            enc_output,enc_hidden = self._encoder(X,initial_state,True)

            dec_input = tf.cast(tf.expand_dims([0]*valid_size,1),tf.float64)

            ## 循环predict
            dec_hidden = enc_hidden
            all_predictions = []
            for t in range(Y.shape[1]):

                predictions,dec_hidden = self._decoder(dec_input,dec_hidden,enc_output,sx = SX,predict = True)

                all_predictions.append(predictions)

                dec_input = predictions

            all_predictions = tf.concat(all_predictions,axis=1)
            # all_predictions = tf.clip_by_value(all_predictions,0,100000)

            Y = unscale_dataset(Y,self._y_mean,self._y_std)
            all_predictions = unscale_dataset(all_predictions,self._y_mean,self._y_std)

            r2 +=  r2_score(Y, all_predictions)

            mae += mean_absolute_error(Y, all_predictions)

            mse += mean_squared_error(Y, all_predictions)

        mae = float('{:.3f}'.format(mae/n_batchs))
        mse = float('{:.3f}'.format(mse/n_batchs))
        r2 = float('{:.3f}'.format(r2/n_batchs))

        return r2,mae,mse

    def predict(self,X,Y):

        valid_size = X.shape[0]

        # 初始化encoder的hideen state
        initial_state = tf.zeros((valid_size,self._units),tf.float64)

        ## 输入encoder
        enc_output,enc_hidden = self._encoder(tf.convert_to_tensor(X,tf.float64),initial_state,True)

        dec_input = tf.cast(tf.expand_dims([0]*valid_size,1),tf.float64)

        ## 循环predict
        dec_hidden = enc_hidden
        all_predictions = []
        for t in range(Y.shape[1]):

            predictions,dec_hidden = self._decoder(dec_input,dec_hidden,enc_output,True)

            all_predictions.append(predictions)

            dec_input = predictions

        all_predictions = tf.concat(all_predictions,axis=1)

        Y = unscale_dataset(Y,self._y_mean,self._y_std)
        all_predictions = unscale_dataset(all_predictions,self._y_mean,self._y_std)

        r2 =  r2_score(Y, all_predictions)

        mae = mean_absolute_error(Y, all_predictions)

        mse = mean_squared_error(Y, all_predictions)

        mae = float('{:.3f}'.format(mae))
        mse = float('{:.3f}'.format(mse))
        r2 = float('{:.3f}'.format(r2))

        return r2,mae,mse,all_predictions
 

def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.div(residual, total))
    return r2

if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(10,10),(10,5),(10,3),(10,1),(5,10),(5,5),(5,3),(5,1),(3,10),(3,5),(3,3),(3,1)]
    for m,n in mn_list:

        # pathObj,m,n,scale = True,feature_set='basic',use_att=False,seperate_static=False,isBidirectional=False,use_l2 = True)
        feature_sets = ['basic-author','basic-structure','basic-author-structure','basic']

        for feature_set in feature_sets:
            for seperate_static in [False]:
                for isBidirectional in [True]:
                    for use_l2 in [True]:
                        for use_att in [True]:
                            s2sm = S2SM(pathObj,m,n,feature_set=feature_set,seperate_static=seperate_static,\
                                isBidirectional=isBidirectional,use_l2=use_l2,use_att=use_att)

                            s2sm.train()