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

from ED_model import gru
from ED_model import Encoder
from ED_model import  loss_function

from ED_model_att import Decoder

from ED_model_att import is_better_result
from functools import partial
from losses import *

from dataset.datasets_construction import construct_RNN_cat_datasets


## Generator
class Generator(tf.keras.Model):
    """docstring for Generator"""
    def __init__(self, units, len_targ):
        super(Generator, self).__init__()
        self._units = units

        self._len_targ = len_targ

        self._encoder = Encoder(self._units)
        ## 不使用attention的decoder
        self._decoder = Decoder(self._units)

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
            # print('shape:{}'.format(predictions.shape))

            all_predictions.append(tf.expand_dims(predictions,1))
            dec_input = predictions

        final_result = tf.concat(all_predictions,1)

        return final_result


## Discriminator对论文的进行encode,然后和预测结果一起进行二类分类
class Discriminator(tf.keras.Model):
    """docstring for Discriminator"""
    def __init__(self, units):
        super(Discriminator, self).__init__(name='discriminator_abc')

        self._units = units
        
        ## encoder对输入进行加权
        self._X_encoder = Encoder(self._units)

        ## 对静态特征进行抽取
        self._static_fc = tf.keras.layers.Dense(self._units,activation='tanh')

        self._static_fc_dropout = tf.keras.layers.Dropout(rate=0.5)

        ## Y同样是序列,使用gru对其加权
        self._Y_encoder = Encoder(self._units)

        ## Dense 根据输出值进行判断
        # self._fc = tf.keras.layers.Dense(self._units,activation='tanh')

        # self._fc_dropout = tf.keras.layers.Dropout(rate=0.5)

        self._fc2 = tf.keras.layers.Dense(2)

    def call(self,dynamic_features,static_features,Y,predict=False):

        # enc_hidden = self._encoder.initialize_hidden_state()
        # print(dynamic_features.shape)
        # print(static_features.shape)
        # print(Y.shape[0])
        initial_state = tf.zeros((Y.shape[0],self._units),tf.float64)
        ## 首先对输入进行encode
        enc_output, enc_hidden = self._X_encoder(dynamic_features,initial_state,predict=predict)

        ## 对静态特征进行向量化
        static_features = self._static_fc(static_features)
        if not predict:
            static_features = self._static_fc_dropout(static_features)

        #使用enc_hidden与static featrues的结果进行concat
        input_vector = tf.concat([enc_hidden,static_features],axis=1)

        ##同样对Y进行向量化
        y_out,y_hidden = self._Y_encoder(tf.expand_dims(Y,2),initial_state,predict=predict)             

        #将X Yvecor进行串联
        vector = tf.concat([input_vector,y_hidden],axis=1)

        ## 判断结果
        # output = self._fc(vector)

        # if not predict:
            # output = self._fc_dropout(output)

        output = self._fc2(vector)

        # print(output.shape)
        return output


class GANCCP:

    def __init__(self,pathObj,m,n):
        ## 加载数据

        scale = False
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,\
        self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,\
        self._valid_Y,self._test_sorted_ids,self._dx_mean,self._dx_std,\
        self._sx_mean,self._sx_std,self._y_mean,self._y_std = construct_RNN_cat_datasets(pathObj,m,n,scale)
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
        # self._n_batchs=10

        ## 数据集
        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        ## 初始化encoder以及decoder
        self._generator = Generator(self._units,self._batch_sz,n)
        self._discriminator = Discriminator(self._units,self._batch_sz)
        self._model_name = 'GANCCP'

        ## pre train
        self._gen_pre_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3,beta_1=0.5,beta_2=0.9)
        # if self._mode=='GAN':

        # else:
        self._dis_pre_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)

        ## optimizer
        self._gen_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)
        self._dis_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.5,beta_2=0.9)

        ## alternative 每一次训练的epoch的大小
        self._n_pre_generator_epoches = 1
        self._n_pre_discriminator_epoches = 1
        self._n_generator_epoches = 1
        self._n_discriminator_epoches = 1

        ## 交叉训练
        self._n_epoches = 50

        ## 记录不同的训练的最佳数值
        self.best_mae = 100
        self.best_mse = 100
        self.best_r2 = 0
        self.best_score = 0

        self._mode = 'WGAN-GP'
        self._penalty_weight = 100

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

    def alternative_training(self):

        # self.reload_latest_checkpoints()
        ## 首先对generator进行预训练
        self.train_generator(pretrain=True)

        ## 使用预训练得到的数据对discriminator进行预训练
        self.train_discriminator(pretrain=True)

        ## 然后进行交叉训练
        for epoch in range(self._n_epoches):

            print('alternative TRAINING EPOCH {} ...'.format(epoch+1))

            ##每次训练都会进行最优化模型的保存
            self.train_generator(pretrain=False)

            ## 每次都会输出准确率计算结果
            self.train_discriminator(pretrain=False)


    def train_generator(self,pretrain = False):

        if pretrain:
            tag = 'pretrain'

            EPOCHS = self._n_pre_generator_epoches

        else:
            tag = 'train'

            EPOCHS = self._n_generator_epoches

        early_stop_count = 0
        # best_mae = 100
        # best_mse = 100
        # best_r2 =0
        # best_score = 0

        train_losses = []
        valid_losses = []
        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()

            # enc_hidden = self._encoder.initialize_hidden_state()

            total_loss = 0

            for (batch,(dynamic_features,static_features,targ)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.train_gen_step(dynamic_features,static_features,targ,pretrain=pretrain)

                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('{}-GENERATOR,sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(tag,self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse,all_predictions = self.gen_predict(self._valid_dynamic_X,self._valid_static_X,self._valid_Y)
            
            train_losses.append(float(total_loss))
            valid_losses.append(float(mae))

            logging.info('{}-Generator,sip-m{}n{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},score:{:.3f},best_score:{:.3f}.'.format(tag,self._m,self._n,epoch+1,total_loss,mae,mse,r2,r2/(mae+mse),self.best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if is_better_result(mae,mse,r2,self.best_mae,self.best_mse,self.best_r2):

                self.best_mae = mae if mae<self.best_mae else self.best_mae
                self.best_mse = mse if mse<self.best_mse else self.best_mse
                self.best_r2 = r2 if r2>self.best_r2 else self.best_r2

                score = r2/(mae+mse)

                if score>self.best_score:
                    self.best_score = score

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
            if early_stop_count>=5:
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

    def train_discriminator(self,pretrain=False):

        ##十分耗时
        all_data,valid_data,test_data = self.gen_data_from_generator()

        # print('Discriminator ...')
        if pretrain:
            tag = 'PRETRAIN'

            n_pretrain_epoches = self._n_pre_discriminator_epoches


        else:
            tag = 'TRAIN'

            n_pretrain_epoches = self._n_discriminator_epoches

        for epoch in range(n_pretrain_epoches):

            start = time.time()

            epoch+=1
            total_loss = 0
            for batch,(dynamic_X,static_X,real_Y,fake_Y) in enumerate(all_data):

                total_loss+=self.train_discriminator_step(dynamic_X,static_X,real_Y,fake_Y,pretrain=pretrain)

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('{}-discriminator,data:sip-m{}n{},model:{},epoch {},batch:{}/{},total average loss:{:.3f}.'.format(tag,self._m,self._n,self._model_name,epoch,batch+1,self._n_batchs,total_loss/(batch+1)))
            
            ##在validation上进行验证
            acc = self.evaluate_discriminator(valid_data)
            print('{}-discriminator data:sip-m{}n{},model:{},epoch {},total average loss:{:.3f},precision on validation:{:.3f},{:.3f},{:.3f}.'.format(tag,self._m,self._n,self._model_name,epoch,total_loss/(batch+1),acc[0],acc[1],acc[2]))

            print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

    def gen_data_from_generator(self):

        ## 生成数据使用保存时候最好的数据
        print('generate fake data from generator ....')

        train_data = []
        ## 使用pretrain的generator进行fake data的生成    
        ## 为了解决vanishing gradients的问题，
        ## 我们将training data里面10%的数据正负颠倒，作为错误噪音
        ## https://medium.com/@jonathan_hui/gan-what-is-wrong-with-the-gan-cost-function-6f594162ce01

        for (batch,(dynamic_X,static_X,real_Y)) in enumerate(self._dataset.take(self._n_batchs)):

            ## generator产生fake out
            fake_Y = self._generator(dynamic_X,static_X,predict=False)

            # print('shape of Y:{}'.format(fake_Y.shape))

            fake_Y = tf.reshape(fake_Y,(fake_Y.shape[0],fake_Y.shape[1]))

            rn = np.random.random()

            ## 加入10%的噪声
            if rn<0:
                batch_data = (dynamic_X,static_X,fake_Y,real_Y)
            else:
                batch_data = (dynamic_X,static_X,real_Y,fake_Y)

            train_data.append(batch_data)

        valid_fake_Y = self._generator(self._valid_dynamic_X,self._valid_static_X,predict=False)

        valid_fake_Y = tf.reshape(valid_fake_Y,(valid_fake_Y.shape[0],valid_fake_Y.shape[1]))

        valid_data = [self._valid_dynamic_X,self._valid_static_X,self._valid_Y,valid_fake_Y]

        test_fake_Y = self._generator(self._test_dynamic_X,self._test_static_X,predict=False)

        test_fake_Y = tf.reshape(test_fake_Y,(test_fake_Y.shape[0],test_fake_Y.shape[1]))

        test_data = [self._test_dynamic_X,self._test_static_X,self._test_Y,test_fake_Y]

        print('data generated done.')

        return train_data,valid_data,test_data

    ## generator pretraining
    def train_gen_step(self,dynamic_X,static_X,targ,pretrain=False):

        with tf.GradientTape() as tape:
            predictions = self._generator(dynamic_X,static_X)

            if pretrain:
                loss = loss_function(tf.expand_dims(targ,2),predictions)
            else:
                ## 如果不是预训练，loss的目标是尽量使discriminator出错
                fake_Y = tf.reshape(predictions,(predictions.shape[0],predictions.shape[1]))
                # print(')
                fake_output = self._discriminator(dynamic_X,static_X,fake_Y)

                ## 为了保证梯度不下降,使用两种loss进行混淆
                loss = generator_loss(fake_output,self._mode)

                
                

            variables = self._generator.trainable_variables

            gradients =  tape.gradient(loss,variables)

            if pretrain:
                self._gen_pre_optimizer.apply_gradients(zip(gradients,variables))
            else:
                self._gen_optimizer.apply_gradients(zip(gradients,variables))

        return loss


    
    def train_discriminator_step(self,dynamic_X,static_X,real_Y,fake_Y,pretrain=False):

        ##获得所有数据之后进行训练
        with tf.GradientTape() as tape:

            ## 使用discriminator进行预测
            fake_output = self._discriminator(dynamic_X,static_X,fake_Y)

            real_output = self._discriminator(dynamic_X,static_X,real_Y)

            dis_loss = discriminator_loss(real_output,fake_output,self._mode)


            ## 为了保证方向的正确性使用mse作为loss
            if self._mode=='GAN':
                dis_loss += loss_function(real_Y,fake_Y)

            if self._mode=='WGAN-GP':

                ##加上GP
                dis_loss += self._penalty_weight*self.gradient_penalty(dynamic_X,static_X,real_Y,fake_Y)

        gradients = tape.gradient(dis_loss,self._discriminator.trainable_variables)


        if pretrain:

            self._dis_pre_optimizer.apply_gradients(zip(gradients,self._discriminator.trainable_variables))

        else:

            self._dis_optimizer.apply_gradients(zip(gradients,self._discriminator.trainable_variables))

        if self._mode=='WGAN':

            [p.assign(tf.clip_by_value(p,-0.1,0.1)) for p in self._discriminator.trainable_variables]

        return dis_loss

    def gradient_penalty(self,dynamic_X,static_X, real, fake):
        ## 这个shape不是real的shape，而是多少维度
        shape = [real.shape[0]]+[1]*(len(real.shape)-1)
        # print(shape)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.,dtype=tf.float64)
        inter = real + alpha * (fake - real)

        with tf.GradientTape() as t:
            t.watch(inter)
            pred = self._discriminator(dynamic_X,static_X,inter)

            grad = t.gradient(pred, inter)
        norm = tf.norm(grad, axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    
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

    ## precision
    def evaluate_discriminator(self,data):
        dynamic_X,static_X,real_Y,fake_Y = data

        fake_output = self._discriminator(dynamic_X,static_X,fake_Y,predict=True)

        real_output = self._discriminator(dynamic_X,static_X,real_Y,predict=True)

        real = tf.concat([tf.zeros_like(fake_output),tf.ones_like(real_output)],axis=0)
        pred = tf.concat([fake_output,real_output],axis=0)

        # print(real.shape)

        # print(real[:10])
        # print(pred[:10])
        # print(pred.shape)

        # acc = self.accuracy(real,pred)

        acc1 = self.accuracy(tf.zeros_like(fake_output),fake_output)

        acc2 = self.accuracy(tf.ones_like(real_output),real_output)

        acc3 = self.accuracy(real,pred)


        # print(acc)

        return acc1,acc2,acc3


    def accuracy(self,real,pred):

        m = tf.keras.metrics.BinaryAccuracy()

        m.update_state(real,pred)

        return m.result().numpy()


        

if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    mn_list=[(5,10),(3,10),(5,1),(5,5),(3,5),(5,3),(3,3),(3,1)]
    # mn_list=[(5,10)]


    for m,n in mn_list:

        ganccp = GANCCP(pathObj,m,n)
        # s2sm.train()

        ganccp.alternative_training()



