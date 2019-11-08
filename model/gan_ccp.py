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

from shallow_regression_model import scale_dataset

tf.keras.backend.set_floatx('float64')
from ED_model import construct_datasets
from ED_model import gru
from ED_model import Encoder
from ED_model import  loss_function

from ED_model_att import BahdanauAttention
from ED_model_att import DecoderAtt

from ED_model_att import is_better_result


## Generator
class Generator(tf.keras.Model):
    """docstring for Generator"""
    def __init__(self, units, batch_sz, len_targ):
        super(Generator, self).__init__()
        self._units = units

        self._batch_sz = batch_sz

        self._len_targ = len_targ

        self._encoder = Encoder(self._units,self._batch_sz)

        self._decoder = DecoderAtt(self._units,self._batch_sz)

    def call(self,dynamic_features,static_features,predict=False):

        # Encoder
        # enc_hidden = self._encoder.initialize_hidden_state()
        initial_state = tf.zeros((dynamic_features.shape[0],self._units),tf.float64)

        enc_output, enc_hidden = self._encoder(dynamic_features,initial_state,predict=predict)

        ## decoder
        dec_input = tf.cast(tf.expand_dims([0]*dynamic_features.shape[0],1),tf.float64)
        dec_hidden = enc_hidden
        all_predictions = []
        for t in range(self._len_targ):

            predictions,dec_hidden = self._decoder(dec_input,dec_hidden,enc_output,static_features,predict=predict)

            all_predictions.append(predictions)
            dec_input = predictions

        return tf.cast(list(zip(*all_predictions)),tf.float64)


## Discriminator对论文的进行encode,然后和预测结果一起进行二类分类
class Discriminator(tf.keras.Model):
    """docstring for Discriminator"""
    def __init__(self, units,batch_sz):
        super(Discriminator, self).__init__()

        self._units = units
        self._batch_sz = batch_sz
        
        ## encoder对输入进行加权
        self._X_encoder = Encoder(self._units,self._batch_sz)

        ## 对静态特征进行抽取
        self._static_fc = tf.keras.layers.Dense(self._units)

        self._static_fc_dropout = tf.keras.layers.Dropout(rate=0.2)

        ## Y同样是序列,使用gru对其加权
        self._Y_encoder = Encoder(self._units,self._batch_sz)

        ## Dense 输出为0,1两个值就行分类
        self._fc = tf.keras.layers.Dense(2)

    def call(self,dynamic_features,static_features,Y,predict=False):

        # enc_hidden = self._encoder.initialize_hidden_state()
        initial_state = tf.zeros((Y.shape[1],self._units),tf.float64)
        ## 首先对输入进行encode
        enc_output, enc_hidden = self._X_encoder(dynamic_features,initial_state,predict=predict)

        ## 对静态特征进行向量化
        static_features = self._static_fc(static_features)
        if not predict:
            static_features = self._static_fc_dropout(static_features)

        #使用enc_hidden与static featrues的结果进行concat
        input_vector = tf.concat([enc_hidden,static_features],axis=1)

        ##同样对Y进行向量化
        y_out,y_hidden = self._Y_encoder(tf.expand_dims(Y,2))             

        #将X Yvecor进行串联
        vector = tf.concat([input_vector,y_hidden],axis=1)

        ## 判断结果
        output = self._fc(vector)


class GANCCP:

    def __init__(self,pathObj,m,n):
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,self._valid_Y,self._test_sorted_ids = construct_datasets(pathObj,m,n)

        ## 构建预训练数据

        print('train model on dataset sip-m{}n{}.'.format(m,n))
        self._m = m
        self._n = n
        self._pathObj = pathObj
        

        ## 超参数
        self._units = 256
        self._batch_sz = 512
        self._buffer_size = len(self._train_Y)
        self._n_batchs =self._buffer_size//self._batch_sz
        # self._n_batchs=40

        ## 数据集
        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        ## 初始化encoder以及decoder
        self._generator = Generator(self._units,self._batch_sz,n)
        self._discriminator = Discriminator(self._units,self._batch_sz)
        self._model_name = 'GANCCP'

        ## pre train
        self._gen_pre_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self._dis_pre_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        ## optimizer
        self._gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self._dis_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)

        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
        self._checkpoint = tf.train.Checkpoint(
            generator = self._generator,
            discriminator = self._discriminator,
            generator_optimizer= self._gen_optimizer,
            discriminator_optimizer = self._dis_optimizer,
            generator_pre_optimizer = self._gen_pre_optimizer,
            discriminator_pre_optimizer = self._dis_pre_optimizer
            )


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

        # pass

    ## generator pretraining

    def pretrain_gen_step(self,dynamic_features,static_features,targ):


        with tf.GradientTape() as tape:
            predictions = self._generator(dynamic_features,static_features)
            loss = loss_function(tf.expand_dims(targ,1),predictions)

            variables = self._generator.trainable_variables

            gradients =  tape.gradient(loss,variables)

            self._gen_pre_optimizer.apply_gradients(zip(gradients,variables))

        return loss


    def pretrain_generator(self):

        EPOCHS = 1000

        early_stop_count = 0
        best_mae = 100
        best_mse = 100
        best_r2 =0
        best_score = 0

        train_losses = []
        valid_losses = []
        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()

            # enc_hidden = self._encoder.initialize_hidden_state()

            total_loss = 0

            for (batch,(dynamic_features,static_features,targ)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.pretrain_gen_step(dynamic_features,static_features,targ)

                total_loss+=batch_loss

                if (batch+1)%10==0 or (batch+1)==self._n_batchs:

                    print('GENERATOR pretraining,sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse,all_predictions = self.gen_predict(self._valid_dynamic_X,self._valid_static_X,self._valid_Y)
            
            train_losses.append(float(total_loss))
            valid_losses.append(float(mae))

            logging.info('sip-m{}n{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},score:{:.3f},best_score:{:.3f}.'.format(self._m,self._n,epoch+1,total_loss,mae,mse,r2,r2/(mae+mse),best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if is_better_result(mae,mse,r2,best_mae,best_mse,best_r2):

                best_mae = mae if mae<best_mae else best_mae
                best_mse = mse if mse<best_mse else best_mse
                best_r2 = r2 if r2>best_r2 else best_r2

                score = r2/(mae+mse)

                if score>best_score:
                    best_score = score

                # self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                r2,mae,mse,all_predictions = self.gen_predict(self._test_dynamic_X,self._test_static_X,self._test_Y)

                logging.info('sip-m{}n{}, saved model, TEST MAE:{}, MSE:{},R2:{}.'.format(self._m,self._n,mae,mse,r2))

                test_result['summary'] = 'sip-m{}n{},{},{},{},{}'.format(self._m,self._n,self._model_name,r2,mae,mse)

                test_result['predictions'] = np.array(all_predictions).tolist()
                early_stop_count=0

            else:
                early_stop_count+=1

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

            ## 如果连续10个回合验证集的loss都不能大于目前保存最好的模型，停止训练
            if early_stop_count>=15:
                break

        ## 将训练过程中loss进行保存
        # loss_obj = {}
        # loss_obj['sip-m{}-n{}'.format(self._m,self._n)] = [list(train_losses),list(valid_losses)]
        # self.save_losses(loss_obj,self._model_name)

        # ## 最好的模型在TEST上面的结果进行保存
        # result_path = self._pathObj.deep_result_prediction_path(self._model_name,self._m,self._n)
        # self._pathObj.save_json(result_path,test_result)

        # ## summary保存到文件里面
        # with open(self._pathObj._deep_result_summary,'a') as f:
        #     summary = test_result['summary']
        #     f.write(summary+'\n')
        #     logging.info('Final performace on test is {}.'.format(summary))


    def save_losses(self,loss_obj,model):
        self._pathObj.save_json(self._pathObj.losses_file(self._m,self._n,model),loss_obj)


    def gen_predict(self,dynamic_X,static_X,Y,isTest=False):

        ## validation set 进行验证
        # self._valid_static_X,self._valid_dynamic_X,self._valid_Y
        all_predictions = self._generator(dynamic_X,static_X,predict=True)
        # print(Y.shape)
        # print(all_predictions.shape)

        all_predictions = tf.reshape(all_predictions,(all_predictions.shape[0],all_predictions.shape[1]))
        r2 =  r2_score(Y, all_predictions)
        # mae = tf.reduce_mean(tf.keras.losses.MAE(Y, all_predictions))
        mae = mean_absolute_error(Y, all_predictions)

        # mse = tf.reduce_mean(tf.keras.losses.MSE(Y, all_predictions))

        mse = mean_squared_error(Y, all_predictions)

        mae = float('{:.3f}'.format(mae))
        mse = float('{:.3f}'.format(mse))
        r2 = float('{:.3f}'.format(r2))

        return r2,mae,mse,all_predictions
        

if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    # mn_list=[(5,10),(3,10),(5,1),(5,5),(3,5),(5,3),(3,3),(3,1)]
    mn_list=[(5,10),(5,5),(5,3)]


    for m,n in mn_list:

        ganccp = GANCCP(pathObj,m,n)
        # s2sm.train()

        ganccp.pretrain_generator()

