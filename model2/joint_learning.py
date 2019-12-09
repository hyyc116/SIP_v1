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
from dataset.datasets_construction import unscale_dataset
from encoder import Encoder
from decoder import Decoder
from joint_model import JointModel
from losses import regress_mse_loss
from losses import classification_loss
from tools import is_better_result


class S2SM:

    def __init__(self,pathObj,m,n,beta1=5,beta2=1):

        self._m = m
        self._n = n
        self._pathObj = pathObj

        scale = True
        ## 加载数据
        self._train_dynamic_X,self._train_static_X,self._train_Y,self._test_dynamic_X,\
        self._test_static_X,self._test_Y,self._valid_dynamic_X,self._valid_static_X,\
        self._valid_Y,self._test_sorted_ids,self._dx_mean,self._dx_std,\
        self._sx_mean,self._sx_std,self._y_mean,self._y_std,\
        self._train_L,self._test_L,self._valid_L= construct_RNN_datasets(pathObj,m,n,scale)

        self._train_X,self._test_X,self._valid_X,self._dx_mean,self._dx_std,\
        self._train_Y,self._test_Y,self._valid_Y,self._y_mean,self._y_std,\
        self._train_L,self._test_L,self._valid_L,\
        self._test_sorted_ids =construct_RNN_datasets(pathObj,m,n,scale)

        ## 数据集
        ## 超参数
        
        self._batch_sz = 512
        self._buffer_size = len(self._train_Y)
        self._n_batchs =self._buffer_size//self._batch_sz

        self._dataset = tf.data.Dataset.from_tensor_slices((self._train_dynamic_X,self._train_static_X,self._train_Y,self._train_L)).shuffle(self._buffer_size)
        self._dataset = self._dataset.batch(self._batch_sz, drop_remainder=True)

        self._test_buffer_size = len(self._test_Y)
        self._n_test_batchs = self._test_buffer_size//self._batch_sz

        self._valid_buffer_size = len(self._valid_Y)
        self._n_valid_batchs = self._valid_buffer_size//self._batch_sz

        self._test_dataset = tf.data.Dataset.from_tensor_slices((self._test_X,self._test_Y)).shuffle(self._test_buffer_size)
        self._test_dataset = self._test_dataset.batch(self._batch_sz, drop_remainder=False)

        self._valid_dataset = tf.data.Dataset.from_tensor_slices((self._valid_X,self._valid_Y)).shuffle(self._valid_buffer_size)
        self._valid_dataset = self._valid_dataset.batch(self._batch_sz, drop_remainder=False)

        
        ## jiont Model根据一般实验的最好的参数进行设置
        self._units = 256
        ## label的数量，0,1,2,3,4,5
        self._vocab_size = 6
        self._use_att= True
        self._dropout_rate=0.5
        self._isBidirectional=True
        ## 初始化encoder以及decoder
        self._model = JointModel(self._units,self._vocab_size,self._use_att,self._dropout_rate,self._isBidirectional)
        self._model_name = 'jont_model'
        ## 正则化的系数
        self._l2_weight = 0.0001
        ## 两个loss之间的关系
        self._beta1 = beta1
        self._beta2 = beta2

        ## optimizer
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        ## 模型的保存位置
        self._checkpoint_dir = './trainning_checkpoints_{}_{}_{}'.format(self._model_name, m,n)
        self._avg_checkpoint_dir = './trainning_checkpoints_avg_{}_{}_{}'.format(self._model_name, m,n)

        self._checkpoint_prefix = os.path.join(self._checkpoint_dir, "ckpt")
        self._trackables = {}
        self._trackables['optimizer']=self._optimizer
        self._trackables['jointmodel']=self._model
        self._checkpoint = tf.train.Checkpoint(**self._trackables)


    def reload_latest_checkpoints(self):
        print('reload latest Checkpoint.....')
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_dir))

    def train_step(self,X,Y,L):

        with tf.GradientTape() as tape:

            regression_result, classification_result = self._model(X,Y,L)

            loss = self._beta1*regress_mse_loss(Y,regression_result)+ self._beta2*classification_loss(L,classification_result)

            batch_loss = (loss/int(Y.shape[1]))

            variables = self._model.trainable_variables

            loss += tf.reduce_mean([tf.nn.l2_loss(var) for var in variables])*self._l2_weight

            gradients =  tape.gradient(loss,variables)

            self._optimizer.apply_gradients(zip(gradients,variables))

        return batch_loss

    def train(self):

        EPOCHS = 1000

        early_stop_count = 0
        best_mae = 100
        best_mse = 100
        best_r2 =0
        best_score = 0

        test_result = {}

        test_result['IDS'] = self._test_sorted_ids

        for epoch in range(EPOCHS):
            start = time.time()


            total_loss = 0

            for (batch,(X,targ,L)) in enumerate(self._dataset.take(self._n_batchs)):

                batch_loss  = self.train_step(X,targ,L)

                total_loss+=batch_loss

                if (batch+1)%50==0 or (batch+1)==self._n_batchs:

                    print('sip-m{}n{}, Epoch {} Batch {}/{} Loss {:.4f}'.format(self._m,self._n,epoch+1,batch+1,self._n_batchs,batch_loss.numpy()))

            total_loss = total_loss/self._n_batchs

            ## 每一个回合结束对模型在valid上面的结果进行评价
            r2,mae,mse,acc = self.predict(self._valid_dataset,self._n_valid_batchs)
            
            logging.info('sip-m{}n{}---beta1:{},beta2:{}, Epoch {}, training Loss {:.4f},validation mae:{}, mse:{},r2:{},acc:{},score:{:.3f},best_score:{:.3f}.'.format(self._m,self._n,self._beta1,self._beta2,epoch+1,total_loss,mae,mse,r2,acc,r2/(mae+mse),best_score))
            ## 每一回合的loss小于best_loss那么就将模型存下来
            ### 在实际的使用中并不能保存下最好的模型，
            ### 我们需要使用三个评价指标共同完成
            ### mae mse的前三位小数相同，并且r2更大
            if epoch>1 and is_better_result(mae,mse,r2,best_score,best_r2,best_mse,best_mae):

                best_mae = mae if mae<best_mae else best_mae
                best_mse = mse if mse<best_mse else best_mse
                best_r2 = r2 if r2>best_r2 else best_r2

                best_score = r2/(mae+mse)

                self._checkpoint.save(file_prefix = self._checkpoint_prefix)
                ## 使用保存的模型对test数据进行验证
                ## 在验证的时候 需要对数据进行unscale
                r2,mae,mse,acc = self.predict(self._dataset,self._n_test_batchs)

                logging.info('sip-m{}n{}, saved model, TEST MAE:{}, MSE:{},R2:{},acc:{}.'.format(self._m,self._n,mae,mse,r2,acc))

                test_result['summary'] = 'sip-m{}n{},beta1:{},beta2:{},{},{},{},{},ACC:{}'.format(self._m,self._n,self._beta1,self._beta2,self._model_name,r2,mae,mse,acc)

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


    def predict(self,dataset,n_batchs):

        r2 = 0
        mae = 0
        mse = 0
        acc = 0
        for (batch,(X,targ,L)) in enumerate(dataset.take(n_batchs)):
            ## validation set 进行验证
            regression_result,classification_result = self._model(dynamic_X,static_X,Y,L,True)

            Y = unscale_dataset(Y,self._y_mean,self._y_std)
            all_predictions = unscale_dataset(regression_result,self._y_mean,self._y_std)


            r2 +=  r2_score(Y, all_predictions)
            mae += mean_absolute_error(Y, all_predictions)
            mse += mean_squared_error(Y, all_predictions)

            m = tf.keras.metrics.SparseCategoricalAccuracy()

            acc += m.result().numpy()

        mae = float('{:.3f}'.format(mae/n_batchs))
        mse = float('{:.3f}'.format(mse/n_batchs))
        r2 = float('{:.3f}'.format(r2/n_batchs))
        acc = float('{:.3f}'.format(acc/n_batchs))

        return r2,mae,mse,acc


if __name__ == '__main__':

    field = 'computer science'
    tag = 'cs'

    pathObj = PATH(field,tag)

    # mn_list=[(5,10),(3,10),(3,5),(5,5),(5,3),(3,3),(5,1),(3,1)(3,10),(3,5),]

    mn_list=[(3,3),(3,1)]
    # mn_list = [(3,3)]
    for m,n in mn_list:

        for beta1,beta2 in [(5,1),(5,5),(1,5),(10,1),(1,10)]:
            s2sm = S2SM(pathObj,m,n,beta1,beta2)
            s2sm.train()

            time.sleep(5)

