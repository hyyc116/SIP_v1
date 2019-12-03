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
from dataset.datasets_construction import construct_RNN_datasets
from dataset.datasets_construction import unscale_dataset


def gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.

    
    return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='tanh', 
                                   recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):

    def __init__(self,enc_units,dropout_rate = 0.5):
        super(Encoder,self).__init__()
        self._enc_units = enc_units
        self._gru = gru(self._enc_units)
        # self._bn = tf.keras.layers.BatchNormalization()

        self._gru_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    ##定义前向传播方法
    def call(self,x,hidden,predict = False):
        
        ## 使用gru的RNN进行前向传播，得到每一步的输出以及state
        output,state = self._gru(x,initial_state = hidden)

        if not predict:
            # output = self._bn(output,training=True)
            output = self._gru_dropout(output)
        # else:
            # output = self._bn(output,training=False)

        return output,state

    def initialize_hidden_state(self,size):
        ## 初始状态是 batch size x enc_units的大小
        return tf.zeros((size,self._enc_units),tf.float64)

class Decoder(tf.keras.Model):

    def __init__(self,dec_units,dropout_rate=0.5):

        super(Decoder,self).__init__()

        self._dec_units = dec_units

        self._gru = gru(self._dec_units)

        # self._bn = tf.keras.layers.BatchNormalization()

        # self._predict = predict

        ## 对静态特征进行抽取
        self._static_fc = tf.keras.layers.Dense(self._dec_units,activation = 'sigmoid')

        self._rnn_dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self._static_fc_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        ## 回归 每一步输出一个数字
        self._fc = tf.keras.layers.Dense(1)


    def call(self,decoder_input,enc_output,static_features,predict=False):

        ## 将static feture的shape从（batch_size,2) 变成（batch_size,units)
        static_features = self._static_fc(static_features)

        if not predict:
            static_features = self._static_fc_dropout(static_features)

        x = tf.concat([enc_output,decoder_input,static_features],axis=-1)

        # print('decoder, x shape {}'.format(x.shape))


        x = tf.expand_dims(x,1)

        # print('decoder, x shape {}'.format(x.shape))

        output,state = self._gru(x)

        if not predict:
            # output = self._bn(output,training=True)
            output = self._rnn_dropout(output)
        # else:
            # output = self._bn(output,training=False)


        output = tf.reshape(output,(-1,output.shape[2]))

        # print(output.shape)

        x = self._fc(output)

        return x,state

    def initialize_hidden_state(self,size):
        return tf.zeros((size,self._dec_units),tf.float64)

# huber = tf.keras.losses.Huber(delta=10)
def loss_function(real,pred):

    if real.shape!=pred.shape:
        print('ERROR: NOT SAME SHAPE IN RESULT.')

    loss = tf.keras.losses.MSE(real, pred)

    # loss = tf.losses.logcosh(real,tru)

    # loss = huber(real,pred)

    return tf.reduce_mean(loss)

class S2SM:

    def __init__(self,pathObj,m,n):

        self._m = m
        self._n = n
        self._pathObj = pathObj

        scale = True
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,\
        self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,\
        self._valid_Y,self._test_sorted_ids,self._dx_mean,self._dx_std,\
        self._sx_mean,self._sx_std,self._y_mean,self._y_std = construct_RNN_datasets(pathObj,m,n,scale)

        ## 数据集
        ## 超参数
        self._units = 256
        self._batch_sz = 512
        self._buffer_size = len(self._train_Y)
        self._n_batchs =self._buffer_size//self._batch_sz

        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        print('train model on dataset sip-m{}n{}.'.format(m,n))


        ## 初始化encoder以及decoder
        self._encoder = Encoder(self._units)
        self._decoder = Decoder(self._units)
        self._model_name = 'ED_model'

        ## optimizer
        # self._optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001,clipvalue=1)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)

        self._avg_checkpoint_dir = './trainning_checkpoints_avg_{}_{}_{}'.format(self._model_name, m,n)


        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")

        self._trackables = {}
        self._trackables['optimizer']=self._optimizer
        self._trackables['encoder']=self._encoder
        self._trackables['decoder']=self._decoder
        self._checkpoint = tf.train.Checkpoint(**self._trackables)


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

        # pass

    def train_step(self,dynamic_features,static_features,targ,enc_hidden):


        with tf.GradientTape() as tape:

            # print('feature shape:{},data type:{}.'.format(dynamic_features.shape,dynamic_features.dtype))

            enc_output, enc_hidden = self._encoder(dynamic_features,enc_hidden)

            ## 需要对enc_output的shape进行查看,并列输入了8种序列特征
            # print('Shape of enc output:{}'.format(enc_output.shape))
            # print('Shape of enc hidden:{}'.format(enc_hidden.shape))

            dec_input = tf.cast(tf.expand_dims([0]*self._batch_sz,1),tf.float64)

            # print('===dec input shape {}'.format(dec_input.shape))

            # print('===dec hidden shape {}'.format(dec_hidden.shape))
            # print('===enc output shape {}'.format(enc_output.shape))

            # print('target shape {},data type {}'.format(targ.shape,targ.dtype))
            # print('targ shape {}'.format(targ.shape))
            loss = 0

            dec_hidden = enc_hidden

            for t in range(0,targ.shape[1]):

                predictions,dec_hidden = self._decoder(dec_input,dec_hidden,static_features)

                # print(targ[:,t].shape,predictions.shape)


                loss += loss_function(tf.expand_dims(targ[:,t],1),predictions)

                rn = np.random.random_sample()

                ##时间t的标准结果作为t+1的x
                # if rn<0.5:
                #     dec_input = tf.expand_dims(targ[:,t],1)
                # else:
                #     ##如果不适用teacher forcing
                dec_input = predictions

            batch_loss = (loss/int(targ.shape[1]))

            variables = self._encoder.trainable_variables + self._decoder.trainable_variables

            gradients =  tape.gradient(loss,variables)

            self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        early_stop_count = 0
        # best_mae = 100
        # best_mse = 100
        # best_r2 =0
        best_score = 0

        train_losses = []
        valid_losses = []
        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self._encoder.initialize_hidden_state(self._batch_sz)

            total_loss = 0

            for (batch,(dynamic_features,static_features,targ)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.train_step(dynamic_features,static_features,targ,enc_hidden)

                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse,all_predictions = self.predict(self._valid_dynamic_X,self._valid_static_X,self._valid_Y)
            
            train_losses.append(float(total_loss))
            valid_losses.append(float(mae))

            logging.info('sip-m{}n{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},score:{:.3f},best_score:{:.3f}.'.format(self._m,self._n,epoch+1,total_loss,mae,mse,r2,r2/(mae+mse),best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if epoch>1 and is_better_result(mae,mse,r2,best_score):

                # best_mae = mae if mae<best_mae else best_mae
                # best_mse = mse if mse<best_mse else best_mse
                # best_r2 = r2 if r2>best_r2 else best_r2

                best_score = r2/(mae+mse)

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                ## 在验证的时候 需要对数据进行unscale
                r2,mae,mse,all_predictions = self.predict(self._test_dynamic_X,self._test_static_X,self._test_Y)

                logging.info('sip-m{}n{}, saved model, TEST MAE:{}, MSE:{},R2:{}.'.format(self._m,self._n,mae,mse,r2))

                test_result['summary'] = 'sip-m{}n{},{},{},{},{}'.format(self._m,self._n,self._model_name,r2,mae,mse)

                test_result['predictions'] = all_predictions.numpy().tolist()
                early_stop_count=0

            else:
                early_stop_count+=1

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

            ## 如果连续10个回合验证集的loss都不能大于目前保存最好的模型，停止训练
            if early_stop_count>=15:
                break

        ## 将训练过程中loss进行保存
        loss_obj = {}
        loss_obj['sip-m{}-n{}'.format(self._m,self._n)] = [list(train_losses),list(valid_losses)]
        self.save_losses(loss_obj,self._model_name)

        ## 最好的模型在TEST上面的结果进行保存
        result_path = self._pathObj.deep_result_prediction_path(self._model_name,self._m,self._n)
        self._pathObj.save_json(result_path,test_result)

        ## summary保存到文件里面
        with open(self._pathObj._deep_result_summary,'a') as f:
            summary = test_result['summary']
            f.write(summary+'\n')
            logging.info('Final performace on test is {}.'.format(summary))


    def save_losses(self,loss_obj,model):
        self._pathObj.save_json(self._pathObj.losses_file(self._m,self._n,model),loss_obj)


    def predict(self,dynamic_X,static_X,Y,isTest=False):

        ## validation set 进行验证
        # self._valid_static_X,self._valid_dynamic_X,self._valid_Y

        valid_size = dynamic_X.shape[0]

        # 初始化encoder的hideen state
        initial_state = tf.zeros((valid_size,self._units),tf.float64)

        ## 输入encoder
        enc_output,enc_hidden = self._encoder(tf.convert_to_tensor(dynamic_X,tf.float64),initial_state,True)

        dec_input = tf.cast(tf.expand_dims([0]*valid_size,1),tf.float64)

        ## 循环predict

        dec_hidden = enc_hidden

        all_predictions = []
        for t in range(Y.shape[1]):

            predictions,dec_hidden = self._decoder(dec_input,dec_hidden,tf.convert_to_tensor(static_X),True)

            all_predictions.append(predictions)

            dec_input = predictions

        all_predictions = tf.concat(all_predictions,axis=1)

        ## 对all_prediction进行unscale

        # print(all_predictions.shape)


        Y = unscale_dataset(Y,self._y_mean,self._y_std)
        all_predictions = unscale_dataset(all_predictions,self._y_mean,self._y_std)

        # print(Y.shape)
        # print(all_predictions.shape)

        r2 =  r2_score(Y, all_predictions)
        # mae = tf.reduce_mean(tf.keras.losses.MAE(Y, all_predictions))

        mae = mean_absolute_error(Y, all_predictions)

        # mse = tf.reduce_mean(tf.keras.losses.MSE(Y, all_predictions))

        mse = mean_squared_error(Y, all_predictions)

        mae = float('{:.3f}'.format(mae))
        mse = float('{:.3f}'.format(mse))
        r2 = float('{:.3f}'.format(r2))

        return r2,mae,mse,all_predictions



def is_better_result(mae,mse,r2,best_score):

    if r2/(mae+mse)>=best_score:
        return True
    else:
        return False


def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.div(residual, total))
  return r2

if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    # mn_list=[(5,10),(3,10),(3,5),(5,5),(5,3),(3,3),(5,1),(3,1)]

    mn_list=[(3,10),(3,5),(3,3),(3,1)]
    # mn_list = [(3,3)]
    for m,n in mn_list:
        s2sm = S2SM(pathObj,m,n)
        s2sm.train()

        time.sleep(5)

